import numpy as np
from PIL import Image, ImageDraw
import cv2
from threading import Thread
import time

from .get_ferret_response import get_ferret_response
from .get_gpt_response import ask_gpt, ask_gpt_3_5

#Split image into num_imgs horizontally
def split_image(image, num_imgs):
    image = np.asarray(image)
    image_width = image.shape[1]
    width_per_img = int(np.ceil(image_width/num_imgs))
    images = []
    image_coords = []
    for i in range(num_imgs):
        coord_dict = dict()
        min_x = i*width_per_img
        max_x = min((i+1)*width_per_img, image_width)
        coord_dict['min_x'] = min_x
        coord_dict['max_x'] = max_x
        coord_dict['min_y'] = 0
        coord_dict['max_y'] = image.shape[0]
        image_coords.append(coord_dict)
        images.append(image[:,min_x:max_x])
    return images, image_coords

def gpt_question(prompt_gpt, views, image_section):
    ans = ask_gpt(prompt_gpt,views)
    for char in ans:
        if char.isdigit():
            image_section[0] = int(char)
    image_section[1] = ans


def ferret_question(prompt_ferret, image, ferret_result):
    answer, boxes, image_with_box = get_ferret_response(prompt_ferret, image)
    ferret_result[0] = boxes
    ferret_result[1] = answer

def draw_view(image, x1, y1, x2, y2):
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1,y1,x2,y2], outline="red")
    image.save("./images/out_gpt.jpg")


def ask_spatial_llm(prompt_ferret, prompt_gpt, image, views, view_coords):
    if prompt_gpt != "":
        image_section = [None, None]
        gpt_thread = Thread(target=gpt_question, args=(prompt_gpt, views, image_section))
        gpt_thread.start()

        ferret_result = [None, None]
        ferret_thread = Thread(target=ferret_question, args=(prompt_ferret, image, ferret_result))
        ferret_thread.start()

        ferret_thread.join()
        gpt_thread.join()
        image_section_num = image_section[0]
        gpt_answer = image_section[1]

        boxes = ferret_result[0]
        ferret_answer = ferret_result[1]

        print("Box = ", boxes)
        if len(boxes) <= 0:
            return gpt_answer
        
        box_centers = []
        for box in boxes:
            box_centers.append([(box[0] + box[2])//2, (box[1] + box[3])//2])   
        return gpt_answer, gpt_answer + "coords=" + str(box_centers)

    else:
        
        ferret_result = [None, None]
        ferret_thread = Thread(target=ferret_question, args=(prompt_ferret, image, ferret_result))
        ferret_thread.start()

        ferret_thread.join()
        
        boxes = ferret_result[0]
        ferret_answer = ferret_result[1]

        
        box_centers = []
        for box in boxes:
            box_centers.append([(box[0] + box[2])//2, image.shape[1] - (box[1] + box[3])//2])
            #since 0 is bottom and it increases as image goes up in unity, flipping y value

        return "coords=" + str(box_centers)