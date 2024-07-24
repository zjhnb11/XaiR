import time
import re
from threading import Thread
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

import gradio as gr

from PIL import Image, ImageDraw, ImageFont

from ..constants import *
from ..utils import Response, Frame, Question
from .llm_querier import LLMQuerier

import cv2

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, value, **kwargs):
        super().__init__(value=value, tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)

class HumanQuerier(LLMQuerier):
    def __init__(self, data_collector):
        super().__init__(data_collector)
        self.current_question: Question = None
        self.serve_thread = Thread(target=self.serve, daemon=True)

        no_image = Image.fromarray(cv2.imread(f"static/images/no-image.jpg"))
        self.image_srcs = [no_image for _ in range(MAX_IMAGES)]
        self.bbox_image_src = None
        self.bbox_image_drawn = False
        self.coords = []
        self.latest_frame = None

    def start(self):
        super().start()
        self.serve_thread.start()

    def build_webapp(self):
        with gr.Blocks(title="Wizard of Oz") as demo:
            gr.Markdown("Images:")
            with gr.Row():
                images = [gr.Image(self.image_srcs[i], label=f"Previous image at time t = {-(MAX_IMAGES-i)+1}") for i in range(len(self.image_srcs)-1)]

            with gr.Row():
                current_image = gr.Image(self.image_srcs[-1], label=f"Current Image")
                sketch_pad = ImageMask(self.image_srcs[-1], label=f"Draw here")
                bbox_image = gr.Image(self.image_srcs[-1], label=f"Current Image with BBOX")

            def draw(inp):
                mask = deepcopy(inp['mask'])
                image = np.array(self.image_srcs[-1].copy())

                colors = ["red"]
                img = Image.fromarray(image)
                draw = ImageDraw.Draw(img)

                mask_new = np.asarray(mask)[:,:,0].copy()
                mask_new = mask_new.astype(np.uint8)
                mask_new = cv2.resize(mask_new, (100, 100), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.coords = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # set to original image size
                    x = x * image.shape[1] // 100
                    y = y * image.shape[0] // 100
                    w = w * image.shape[1] // 100
                    h = h * image.shape[0] // 100
                    center = [x + w//2, y + h//2]
                    self.coords.append(center)

                    bbox = [x, y, x+w, y+h]
                    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=colors[0], width=4)

                if len(self.coords) > 0:
                    self.bbox_image_drawn = True
                    self.bbox_image_src = np.array(img).copy()

            sketch_pad.edit(draw, inputs=[sketch_pad], outputs=None, queue=True)

            update_image_button = gr.Button("Update Center Image")

            def update_image():
                return self.image_srcs[-1]

            update_image_button.click(update_image, outputs=[sketch_pad])

            user_question = gr.Textbox("No question", label="User Question:")

            gr.Markdown("Start typing below and then press **Enter** to send the output.")
            with gr.Row():
                response_input = gr.Textbox(placeholder="Input your response here", label="Response:")
                instruction_status = gr.Radio(["True", "False"], label="Instruction Status", info="Is the instruction done?")

            def update_response(response_input, instruction_status):
                self.set_answer(response_input, instruction_status)
                self.current_question = None
                return gr.update(value="")

            response_input.submit(update_response, inputs=[response_input, instruction_status], outputs=[response_input])

            send_button = gr.Button("Send")
            send_button.click(update_response, inputs=[response_input, instruction_status], outputs=[response_input])

            def update_images():
                user_question_text = "No new question yet... This textbox will update with the latest question."
                if self.current_question is not None:
                    user_question_text = self.current_question.text

                if not self.bbox_image_drawn:
                    self.bbox_image_src = self.image_srcs[-1].copy()

                return [user_question_text] + [img for img in self.image_srcs[:-1]] + [self.image_srcs[-1], self.bbox_image_src]

            demo.load(update_images,
                      inputs=None,
                      outputs=[user_question] + [img for img in images] + [current_image, bbox_image],
                      every=0.1)

        return demo

    def serve(self):
        webapp = self.build_webapp()
        webapp.queue(api_open=False).launch()

    def process_question_and_frame(self, question_to_process, frame_to_process):
        self.current_question = question_to_process

        # self.bbox_image_drawn = False
        for frame in self.data_collector.get_latest_n_frames(5):
            self.image_srcs.append(frame.img)
            self.image_srcs.pop(0)
            self.latest_frame = frame

    def get_latest_frameID(self):
        return self.latest_frame.frameID

    def set_answer(self, answer_text, instruction_status):
        if self.current_question is not None:
            if len(self.coords) > 0:
                # add punctuation if needed
                if re.match('^[A-Z][^?!.]*[?.!]$', answer_text) is None:
                    answer_text += "."
                answer_text += f"coords={self.coords}"
            self.answer = Response(instruction_status + ". " + answer_text, self.current_question)
