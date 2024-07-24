import argparse
import time
import ssl
import cv2
from threading import Thread

from aiohttp import web

from .constants import *
from .data_collector import DataCollector
from .model_interface.ferret_gpt_querier import FerretGPTQuerier
from .model_interface.human_querier import HumanQuerier
from .utils import Frame, Question

import av.logging
# monkey patch av.logging.restore_default_callback, ie remove the annoying ffmpeg prints
restore_default_callback = lambda *args: args
av.logging.restore_default_callback = restore_default_callback
av.logging.set_level(av.logging.ERROR)


data_collector = DataCollector(MAX_IMAGES)
ferret_gpt = FerretGPTQuerier(data_collector)
human = HumanQuerier(data_collector)


def process_user_input():
    frameID = 0

    while True:
        user_input = input()
        user_input = user_input.strip()
        if user_input:
            last_space_index = user_input.rfind(' ')
            user_question, user_image_path = user_input[:last_space_index], user_input[last_space_index+1:]
            user_image_path = user_image_path.replace('\'', '')
            print('User input:', user_question, "| User image path:", user_image_path)
            print()

            try:
                img = cv2.imread(user_image_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if img is not None:
                    question_to_ask = Question(user_question)
                    data_collector.add_question(question_to_ask)

                    frame_to_add = Frame(img, frameID)
                    data_collector.add_frame(frame_to_add)

                    frameID += 1
            except Exception as e:
                print('Error:', e)

        time.sleep(0.1)

def print_answer():
    while True:
        if human.get_answer() is not None:
            response = human.get_answer()
            print(f'Answer from human: \"{response.text}\"')
            print('Time taken for human:', response.timestamp - response.question.timestamp, 'seconds')
            print()
            human.clear_answer()

        if ferret_gpt.get_answer() is not None:
            response = ferret_gpt.get_answer()
            print(f'Answer from LLM assistant: \"{response.text}\"')
            print('Time taken for LLM:', response.timestamp - response.question.timestamp, 'seconds')
            print()
            ferret_gpt.clear_answer()

        time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cognitive Assistant Server with manual input'
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

    if args.cert_file:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    ferret_gpt.start()
    human.start()

    print_answer_thread = Thread(target=print_answer, daemon=True)
    print_answer_thread.start()

    process_user_input()
