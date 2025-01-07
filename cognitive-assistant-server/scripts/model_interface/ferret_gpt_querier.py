import time

from ..utils import Response
from .llm_querier import LLMQuerier
from .ferret_with_gpt import ask_spatial_llm, split_image, ask_gpt

class FerretGPTQuerier(LLMQuerier):
    def process_question_and_frame(self, question_to_process, frame_to_process):
        frame_to_give_llm = frame_to_process.img
        # question_to_ask_ferret = \
        #     'You are a helpful AI assistant to a human. ' + \
        #     'For context, the input image is taken from a head mounted camera. ' + \
        #     'Give your answer in just one sentence, as conscise as possible. ' + \
        #     'After answering the question in a sentence, if the object described in the question is present, just give me the location of the object and if not, just say "not present" and nothing else.' + \
        #     'User Question: ' + question_to_process.text
        question_to_ask_gpt = \
            'You are a helpful AI assistant to a human. ' + \
            'For context, the images are of the space around me, as taken from a head mounted camera. ' + \
            'Give your answer in just one sentence, as conscise as possible. ' + \
            'Answer the question in just one or two sentences ' +\
            'User Question: ' + question_to_process.text

        views, view_coords = split_image(frame_to_give_llm,1)
        # response_text = ask_spatial_llm(question_to_ask_ferret, question_to_ask_gpt, frame_to_give_llm, views, view_coords)
        response_text = ask_spatial_llm(
        prompt_ferret="",           # 或者随便传个空字符串
        prompt_gpt=question_to_ask_gpt,
        image=frame_to_give_llm,
        views=views
    )
        self.answer = Response(response_text, question_to_process)
