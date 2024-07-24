import time
from threading import Thread, Event
from queue import Queue
import numpy as np

from ..utils import Response
from .ferret_with_gpt import ask_spatial_llm
from .instructions_prepare import instruction_breakdown, get_curr_instruction, is_instruction_complete

class TutorialFollower:
    def __init__(self, data_collector):
        self.data_collector = data_collector

        self.tutorial_instruction_processing = Thread(target=self._thread_loop, daemon=True)
        self.answer: Response = None
        self.current_instruction = ""
        self.current_instruction_index = 0
        self.task = "multimeter" #"humidifier"

    def get_inst(self, instruction_file, input_file):
        file = open(instruction_file, "r")
        text = file.read()
        self.instructions = text.splitlines()
        self.instructions.append("Task completed!")
        print(self.instructions)
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
            latest_frames = self.data_collector.get_latest_n_frames(2)
            frame_img = []
            if (len(latest_frames)>0):
                last_frameID = latest_frames[-1].frameID
                for frame in latest_frames:
                    frame_img.append(np.asarray(frame.img))
                print("here")
                try:
                    answer = is_instruction_complete(frame_img, self.instructions, self.current_instruction)
                except:
                    continue
                print(answer)
                self.prev_instruction_index =  self.current_instruction_index
                for line in answer.splitlines():
                    if "true" in line.lower():
                        self.current_instruction_index += 1
                    if 'Instruction number:' in line:
                        current_instruction_index = line.split('Instruction number:')[1]
                        self.current_instruction_index = int(current_instruction_index) 
                    
                
                if  self.current_instruction_index < self.prev_instruction_index:
                    self.current_instruction_index = self.prev_instruction_index
                self.current_instruction = self.instructions[self.current_instruction_index]
                self.answer = self.current_instruction + '\n Current instruction state: ' + answer
                if (self.prev_instruction_index != self.current_instruction_index) or self.current_instruction_index == 0:
                    ferret_prompt = "Find the following object for me. Give me an answer as a comma separated list with the format: object name = <co-ordinates>, object name:<co-ordinates>. If you cannot find a certain object with confidence, replace its coordinates with None. Do not give me more coordinates than I have asked for. Only give me coordinated for the objects I asked. The objects are: " + str(self.inst_inputs[self.current_instruction_index])
                    response = ask_spatial_llm(ferret_prompt, "",frame_img[-1], None, None)
                    print("Ferret response = " + response)
                    self.answer = str(last_frameID)+ "///" + self.current_instruction + '\n Current instruction state: ' + answer + response
            time.sleep(0.1)


    def get_answer(self):
        return self.answer

    def clear_answer(self):
        self.answer = None
