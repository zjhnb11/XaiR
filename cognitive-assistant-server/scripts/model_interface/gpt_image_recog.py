from openai import OpenAI
import os
from PIL import Image
import io
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from scripts.constants import API_KEY



import base64

# Function to encode an image file to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        b = image_file.read()
        image = Image.open(io.BytesIO(b))
        image.save("later.png")
        encoded_string = base64.b64encode(b).decode('utf-8')
    return encoded_string

def create_question_content(prompt, views):
    content_list = []

    temp = dict()
    for view in views:
        cv2.imwrite("temp.png", cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
        temp["type"] = "image"
        temp["image"] = encode_image_to_base64("temp.png")
        content_list.append(temp.copy())

    text = dict()
    text["type"] = "text"
    text["text"] = prompt
    content_list.append(text.copy())

    return content_list


def ask_gpt(prompt, views):
    content = create_question_content(prompt, views)
    client = OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=3500,
    )
    response = response.choices[0]
    return response.message.content

def ask_gpt_3_5(prompt):
    client = OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                    ]
            }
        ],
        max_tokens=4000,
    )
    response = response.choices[0]
    return response.message.content
