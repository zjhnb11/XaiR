import time
from threading import Thread, Event
from queue import Queue
import numpy as np
from PIL import Image
import numpy as np

from ..utils import Response, Question

class TutorialFollowerHuman:
    def __init__(self, data_collector, querier):
        self.data_collector = data_collector
        self.querier = querier
        self.tutorial_instruction_processing = Thread(target=self._thread_loop, daemon=True)
        self.answer: Response = None
        self.current_instruction = ""
        self.current_instruction_index = 0
        self.task = "multimeter" #"humidifier", "blocks_1", "blocks_2", "blocks_3", "multimeter"

    def get_inst(self, instruction_file, input_file):
        file = open(instruction_file, "r")
        text = file.read()
        self.instructions = text.splitlines()
        self.instructions.append("Task completed!")
        file = open(input_file, "r")

        text = file.read()
        self.inst_inputs = text.splitlines()

    def start(self):
        if self.task == "humidifier":
            self.get_inst("/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/humidifier/ground_truth.txt","/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/humidifier/ground_truth_inputs.txt")
        elif self.task == "blocks_1":
            self.get_inst("/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/building_blocks/instructions_1.txt", "/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/building_blocks/inputs_1.txt")
        elif self.task == "blocks_2":
            self.get_inst("/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/building_blocks/instructions_2.txt", "/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/building_blocks/inputs_2.txt")
        elif self.task == "blocks_3":
            self.get_inst("/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/building_blocks/instructions_3.txt", "/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/building_blocks/inputs_3.txt")
        elif self.task == "multimeter":
            self.get_inst("/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/multimeter/instructions.txt", "/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/multimeter/inputs.txt")
        
        self.current_instruction = self.instructions[0]
        
        self.tutorial_instruction_processing.start()

    def _thread_loop(self):
        while True:
            if len(self.data_collector.get_latest_n_frames(5)) < 5:
                question_to_ask = Question("User instruction: "+ self.instructions[self.current_instruction_index])
                self.data_collector.add_question(question_to_ask)
            
            if self.querier.get_answer() is not None:
                answer = self.querier.get_answer().text
                last_frameID = self.querier.get_latest_frameID()
                self.querier.clear_answer()
                if "true" in answer.lower():
                        self.current_instruction_index += 1
                self.answer = str(last_frameID)+ "///" + self.current_instruction + '\n Current instruction state: ' + answer
                question_to_ask = Question("User instruction: "+ self.instructions[self.current_instruction_index])
                self.data_collector.add_question(question_to_ask)

            time.sleep(0.1)


    def get_answer(self):
        return self.answer

    def clear_answer(self):
        self.answer = None
