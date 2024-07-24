import time
from threading import Thread, Event
from queue import Queue

from ..utils import Response

class LLMQuerier:
    def __init__(self, data_collector):
        self.data_collector = data_collector

        self.spatial_thread = Thread(target=self._thread_loop, daemon=True)
        self.answer: Response = None

    def start(self):
        self.spatial_thread.start()

    def _thread_loop(self):
        while True:
            # wait for new question
            self.data_collector.wait_for_question()

            question_to_process, latest_frame = self.data_collector.get_latest_query()
            if question_to_process is not None and latest_frame is not None:
                self.process_question_and_frame(question_to_process, latest_frame)

            self.data_collector.clear_new_question_event()

            time.sleep(0.1)

    def process_question_and_frame(self, question_to_process, frame_to_process):
        raise NotImplementedError

    def get_answer(self):
        return self.answer

    def clear_answer(self):
        self.answer = None
