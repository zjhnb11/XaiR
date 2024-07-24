from openai import OpenAI
from PIL import Image
import base64
import time
import cv2
from .ferret_with_gpt import ask_gpt, ask_gpt_3_5
from scripts.constants import API_KEY

# file = open("/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/overall_inst.txt", "r")

# text = file.read()
# instructions = text.splitlines()

#OpenAI API Key
client = OpenAI(api_key=API_KEY)

# Function to encode an image file to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

#Break instructions down into bite size steps
def instruction_breakdown(instructions):
    prompt = "Can you break these intructions down into detailed steps of how to do each of the instruction? Assume all the components needed for the instructions are available. You do not need to give me too obvious or overly simplified steps. Give me each step as a bullet point list. Do not add extra lines or titles or details. If an instruction gives an option of two methods, just pick one and make sure the rest of the ingredients are consistant with it. Do not give me extra information or optional steps. Instructions: "+ instructions
    return ask_gpt_3_5(prompt).splitlines()
    

def get_curr_instruction(frames, instructions):
    #check whether instruction has been done
    # frames = ["/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test130.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test132.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test134.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test136.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test138.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test140.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test142.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test144.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test146.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test148.png"]
    prompt = "Provided is a list of instructions to perform a task. Look at the ego-centric images that show the last 10 seconds of what I have been doing from my headmounted device and tell me which step I should do next, that is, what is the instruction I should currently follow. Give me the instruction as an instruction number and nothing else in the format: 'Instruction number: <instruction>', with the first instruction being instruction 1. If you don't have an answer, answer with instruction 1. Make this inferrence based on what you see as the state of my environment in the image. Also tell me what objects in my image I need to do this instruction. Tell me in the format: 'Needed objects: <list of objects>'. Here are the instructions:" + str(instructions)
    return ask_gpt(prompt, frames)
    # content_list = []
    # for frame in frames:
    #     temp = dict()
    #     cv2.imwrite("temp1.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #     temp["type"] = "image"
    #     temp["image"] = encode_image_to_base64("temp1.png")
    #     content_list.append(temp.copy())

    # text = dict()
    # text["type"] = "text"
    # text["text"] = 
    # content_list.append(text.copy())

    # response = client.chat.completions.create(
    #     model="gpt-4-vision-preview",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": content_list
    #         }
    #     ],
    #     max_tokens=4000,
    # )
    # response = response.choices[0]
    # return response.message.content

def is_instruction_complete(frames, instructions, current_instruction):
    #check whether instruction has been done
    # frames = ["/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test130.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test132.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test134.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test136.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test138.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test140.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test142.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test144.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test146.png",
    #                     "/home/ssrinidh/Sruti/cognitive-assistant/results/making coffee/frames/test148.png"]
    prompt = "I am currently trying to do the instruction: " + current_instruction + "\n Have I done the instruction? I am giving you a frame showing the current state of my environment from an ego-centric view. Does it look like the instruction may have been done? If there is any chance it might be done, say true. Be linient in your responses. Answer just True or False. If false, tell me what I am missing. If unsure, say 'true'. Remember right is left and left is right (the image is mirrored). Here is the complete list of instructions: " + str(instructions)
    #I am giving you the frames for the last 10 seconds of me doing the task, where each frame is 1 second apart
    return ask_gpt(prompt, frames)
    #