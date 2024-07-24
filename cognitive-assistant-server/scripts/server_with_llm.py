import argparse
import asyncio
import json
import logging
import os
import ssl
import time
import uuid

import whisper
import cv2
import torch
import numpy as np

from datetime import datetime, timedelta
from threading import Thread, Event
import speech_recognition as sr
from PIL import Image
from queue import Queue

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import AudioFrame, AudioFormat, AudioResampler, AudioFifo

from jinja2 import Environment, FileSystemLoader

from .constants import *
from .data_collector import DataCollector
from .model_interface.ferret_gpt_querier import FerretGPTQuerier
from .spatial_memory import HistoryLogger
from .utils import Frame, Question

import av.logging
# monkey patch av.logging.restore_default_callback, ie remove the annoying ffmpeg prints
restore_default_callback = lambda *args: args
av.logging.restore_default_callback = restore_default_callback
av.logging.set_level(av.logging.ERROR)


######################## INITIALIZATION ############################################


logger = logging.getLogger('pc')
pcs = set()
audio_relay = MediaRelay()


data_collector = DataCollector(MAX_IMAGES)
# history = HistoryLogger(data_collector)
ferret_gpt = FerretGPTQuerier(data_collector)


# shared variables
datachannels = {}

frameID = None
sessionId = 0


# def scene_description_background_processing():
#     global latest_frames_queue, history
#     video_frames = []
#     frame_count = 0
#     first_frame = None
#     while True:
#         print("history =  ", history.session_started)
#         if history.session_started:
#             frame_to_process = None

#             if not latest_frames_queue.empty():
#                 for i in range(min(10,len(latest_frames_queue.queue))):
#                     frame_to_process = latest_frames_queue.queue[0] # just get the first frame
#                     latest_frames_queue.queue.popleft()
#                     if (frame_to_process.frameID >= history.startFrame) and ((history.endFrame == None) or (frame_to_process.frameID <= history.endFrame)):
#                         if frame_to_process is not None:

#                             if frame_count == 0:
#                                 first_frame = frame_to_process
#                             frame_to_give_llm = frame_to_process.img
#                             video_frames.append(np.asarray(frame_to_give_llm))
#                             frame_count += 1
#                         if frame_count == 10:
#                             question_to_ask = \
#                                 'You are a helpful AI assistant to a human. ' + \
#                                 'For context, the images are of the space around me, as taken from a head mounted camera. ' + \
#                                 'Give your answer in just one sentence, as conscise as possible. ' + \
#                                 'This is a series of frames from a segment of a video where each frame is one second apart. Can you tell me what is happening in this segment of a video? Give me a two sentence summary'
#                             #'After answering the question in a sentence, tell me which image contains the object described in the question? Just give me the image number, and if the object is not present, say "not present"' + \
#                             #The overall video is about how to use a humidifier.
#                             start = time.time()
#                             response = ask_gpt(question_to_ask, video_frames)
#                             history.add_scene_description(first_frame.timestamp, first_frame.frameID, response)
#                             end = time.time()
#                             print('Time taken for LLM response: ', end - start, ' seconds')
#                             print('\n===\nScene Description:\n', response, '\n===\n')
#                             video_frames = []
#                             frame_count = 0
#                     elif ((history.endFrame != None) and (frame_to_process.frameID > history.endFrame)):
#                         history.session_finish_processing()
#                         break

#         elif (not pending_questions_queue.empty()) or (history.session_started):
#             latest_frames_queue.queue.clear()
#         time.sleep(6)


#############################################################################################################################


class WebRTCSource(sr.AudioSource):
    def __init__(self, sample_rate=None, chunk_size=1024, sample_width=4):
        # Those are the only 4 properties required by the recognizer.listen method
        self.stream = WebRTCSource.MicrophoneStream()
        self.SAMPLE_RATE = sample_rate  # sampling rate in Hertz
        self.CHUNK = chunk_size  # number of frames stored in each buffer
        self.SAMPLE_WIDTH = sample_width  # size of each sample

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stream.close()
        finally:
            self.stream = None

    class MicrophoneStream(object):
        def __init__(self):
            self.stream = AudioFifo()
            self.event = Event()

        def write(self, frame: AudioFrame):
            assert type(frame) is AudioFrame, 'Tried to write something that is not AudioFrame'
            self.stream.write(frame=frame)
            self.event.set()

        def read(self, size) -> bytes:
            frames: AudioFrame = self.stream.read(size)

            # while no frame, wait until some is written using an event
            while frames is None:
                self.event.wait()
                self.event.clear()
                frames = self.stream.read(size)

            # convert the frame to bytes
            data: np.ndarray = frames.to_ndarray()
            return data.tobytes()


def process_user_speech_thread(source: sr.AudioSource, model_name="small", record_timeout=4, phrase_timeout=3, energy_threshold=100):
    global pending_questions, sessionId, history

    audio_model = whisper.load_model(model_name + ".en")
    data_queue = Queue()
    phrase_time = None

    transcription = ['']

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = energy_threshold
    recognizer.dynamic_energy_threshold = False

    recognizer.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    recognizer.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    while True:
        now = datetime.now(datetime.UTC)
        start = time.time()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty():
            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                phrase_complete = True
            # This is the last time we received new audio data from the queue.
            phrase_time = now

            # Combine audio data from queue
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            # Convert in-ram buffer to something the model can use directly without needing a temp file.
            # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
            # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Read the transcription.
            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text = result['text'].strip()

            # If we detected a pause between recordings, add a new item to our transcription.
            # Otherwise edit the existing one.
            if phrase_complete:
                transcription.append(text)
            else:
                transcription[-1] = text

            line = transcription[-1].lower()
            if len(line) > 0 :
                print('\n===\nUser question:\n', line, '\n===\n')
                print(line.split(' ')[0])
                if 'alexa' in line.split(' ')[0]:
                    line = line.split('alexa')[1]
                    print("here")
                    if (("start" in line) and ("recording" in line)):
                        print("here again")
                        history.start(sessionId, frameID)
                        print("started recording")
                        sessionId += 1
                    elif (("stop" in line) and ("recording" in line)):
                        print("here again")
                        history.stop(frameID)
                        print("stopped recording")
                        sessionId += 1
                    else:
                        question_to_ask = Question(line)
                        data_collector.add_question(question_to_ask)

                        print("Time to process user audio: ", time.time() - start, "seconds")

        time.sleep(0.1)


class AudioTransformTrack(MediaStreamTrack):
    kind = 'audio'

    def __init__(self, track):
        super().__init__()
        self.track = track
        rate = 16_000  # Whisper has a sample rate of 16000
        audio_format = 's16p'
        sample_width = AudioFormat(audio_format).bytes
        self.resampler = AudioResampler(format=audio_format, layout='mono', rate=rate)
        self.source = WebRTCSource(sample_rate=rate, sample_width=sample_width)

    async def recv(self):
        global answer_from_assistant

        if ferret_gpt.get_answer() is not None:
            response = ferret_gpt.get_answer()
            datachannels[LLM_OUTPUT_DC_LABEL].send(response)
            ferret_gpt.clear_answer()

            print(f'Answer from LLM assistant: \"{response}\"')
            print('Time taken:', response.time_taken, 'seconds')

        out_frame: AudioFrame = await self.track.recv()

        out_frames = self.resampler.resample(out_frame)

        for frame in out_frames:
            self.source.stream.write(frame)

        return out_frame


async def receiveImage(request):
    global frameID
    # logger.info("trying to read")
    request.auto_decompress = False
    image = b''
    while True:
        chunk = await request.content.read(1024)  # Adjust the buffer size as needed
        if not chunk:
            break
        image += chunk
    logger.info(len(image))
    frameID = image[:4]
    frameID = int.from_bytes(frameID, byteorder='little')
    logger.info(frameID)
    img = Image.frombytes('RGBA', (640,480), image[4:], 'raw')
    data_collector.add_frame(Frame(img, frameID))
    # latest_frames_queue.put(Frame(img, frameID))
    img.save("test" + str(frameID)+ ".png")
    # logger.info("got image")


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

    pc = RTCPeerConnection()
    pc_id = 'PeerConnection(%s)' % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + ' ' + msg, *args)

    log_info('Created for %s', request.remote)

    @pc.on('datachannel')
    def on_datachannel(channel):
        global datachannels
        if channel.label == LLM_OUTPUT_DC_LABEL:
            datachannels[LLM_OUTPUT_DC_LABEL] = channel

        # @channel.on("message")
        # def on_message(message):
        #     log_info("IT RECEIVESSSSSSSSSSSSSSSSSSSSSS")

    @pc.on('connectionstatechange')
    async def on_connectionstatechange():
        log_info('Connection state is %s', pc.connectionState)
        if (pc.connectionState == 'closed'):
            while(len(data_collector.latest_frames_queue.queue) != 0):
                continue
            history.print_history()
        if pc.connectionState == 'failed':
            await pc.close()
            pcs.discard(pc)

    @pc.on('track')
    def on_track(track):
        log_info('Track %s received', track.kind)

        if track.kind == 'audio':
            audio_track = audio_relay.subscribe(track)
            if audio_track:
                t = AudioTransformTrack(audio_track)
                pc.addTrack(t)
                thread = Thread(target=process_user_speech_thread, daemon=True, args=(t.source,))
                thread.start()

        @track.on('ended')
        async def on_ended():
            log_info('Track %s ended', track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps(
            {'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type}
        ),
    )


async def index(request):
    image_urls = get_image_files()
    user_query = get_user_query()

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('index_template.html')

    html = template.render(image_urls=image_urls, user_query=user_query)

    return web.Response(text=html, content_type='text/html')


def get_user_query():
    latest_question = data_collector.get_latest_question()
    if latest_question is not None:
        return latest_question.text
    return '<No question yet>'


def get_image_files():
    frames = data_collector.get_latest_n_frames(MAX_IMAGES)
    image_urls = []
    for frame in frames:
        image_path = IMAGE_FOLDER + str(frame.frameID) + '.jpg'
        cv2.imwrite(image_path, cv2.cvtColor(frame.img, cv2.COLOR_BGR2RGB))
        image_urls.append(image_path)
    while len(image_urls) < MAX_IMAGES:
        image_urls.append(IMAGE_FOLDER + 'no-image.jpg')
    return image_urls[::-1]


async def handle_form(request):
    return web.HTTPFound('/')


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WebRTC audio / video / data-channels demo'
    )
    parser.add_argument('--cert-file', help='SSL certificate file (for HTTPS)')
    parser.add_argument('--key-file', help='SSL key file (for HTTPS)')
    parser.add_argument(
        '--host', default='0.0.0.0', help='Host for HTTP server (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', type=int, default=8000, help='Port for HTTP server (default: 8000)'
    )
    parser.add_argument('--verbose', '-v', action='count')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    ferret_gpt.start()

    # history_logging_thread = Thread(target=scene_description_background_processing)
    # history_logging_thread.start()


    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index)
    app.router.add_post('/submit', handle_form)
    app.router.add_static('/static/', 'static')
    app.router.add_post('/offer', offer)
    app.router.add_put('/image', receiveImage)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
