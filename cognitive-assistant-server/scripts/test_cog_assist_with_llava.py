from .model_interface.ferret_with_gpt import ask_spatial_llm
from .model_interface.gpt_image_recog import ask_gpt, ask_gpt_3_5
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
from tqdm import tqdm

from .spatial_memory import HistoryLogger

if __name__ == '__main__':
    folders = ['/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/video_segments/humidifier/', '/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/video_segments/coffee/', '/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/video_segments/sandwich/', '/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/video_segments/setting table/', '/home/ssrinidh/Sruti/cognitive-assistant/results/Ego-centric Videos/video_segments/soldering/']
    intent = ["set up a humidifier.", "make coffee.", "make a sandwich.", "set up a dinner table.", "solder."]
    for i in tqdm(range(5)):
        for rep in tqdm(tqdm(range(10))):
            file = open(folders[i] + "overall_inst_%d.txt"%rep, "r")
            scene_desc = file.read()
            prompt = 'I took a video of me trying to' + intent[i] +'. I gave an AI model the video in several' +\
                    'segments and asked it to describe what I was doing. It is not the best model' +\
                    ' but it got the general idea. Here are the descriptions of each video segment it' +\
                    ' came up with. Based on this, can you give me a concise and practical list of' +\
                    ' instructions to do what I am doing in the video? Each line is a desription of each' +\
                    ' segment. Answer based on what i did in the video only and not your own knowledge. Do not add any extra steps that has not been done in the video: \n' + scene_desc
            answer = ask_gpt_3_5(prompt)
            print(answer)
            with open(folders[i] + "llava_inst_%d.txt" % rep, "w") as outfile:
                outfile.write(answer)
        