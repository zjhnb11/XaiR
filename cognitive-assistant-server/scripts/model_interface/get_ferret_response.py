import requests
import time
import json
import base64
import cv2
import numpy as np
import io

from PIL import Image, ImageDraw

worker_addr = "http://localhost:10000"

headers = {"User-Agent": "FERRET Client"}

input_prompt = "A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0, 0], bottom-right [width-1, height-1]. Increasing x moves right; y moves down. Bounding box: [x1, y1, x2, y2]. Image size: 1000x1000. Follow instructions.  USER: <image>"

def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', np.asarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
    encoded_image = base64.b64encode(buffer)
    image = Image.open(io.BytesIO(buffer))
    return encoded_image.decode('utf-8')

def gen_ferret_prompt(prompt):
    return input_prompt + "\n" + prompt + " ASSISTANT:"

def ask_ferret(prompt, img, model_name="ferret-13b-v1-3", temperature=0.2, top_p=0.7, max_new_tokens=512, stop_token="</s>"):
    pload = {
        "model": model_name,
        "prompt": gen_ferret_prompt(prompt),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": stop_token,
        "images": [encode_image_to_base64(img)],
    }

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(input_prompt):].strip()
                    yield True, output
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    yield False, output
                time.sleep(0.01)
    except requests.exceptions.RequestException as e:
        print("Error: ", e)
        yield False, str(e)

def get_coordinates_from_prompt(answer_prompt):
    answers = answer_prompt.split("[")
    box_coords = []
    for ans in answers[1:]:
        [box_text, _] = ans.split("]")
        indv_coords = box_text.split(',')
        coords = []
        for c in indv_coords:
            coords.append(int(c))
        box_coords.append(coords)
    return(box_coords)

def extract_boxes(answer):
    [before, after] = answer.split("ASSISTANT: ")
    answer_prompt = after.split('"')
    answer_prompt = answer_prompt[0]
    boxes = get_coordinates_from_prompt(answer_prompt)
    return answer_prompt, boxes

def resize_bbox(boxes, image_w=None, image_h=None, default_wh=1000):
    ratio_w = image_w * 1.0 / default_wh
    ratio_h = image_h * 1.0 / default_wh
    new_boxes = []
    for box in boxes:
        new_box = [int(box[0] * ratio_w), int(box[1] * ratio_h), \
                int(box[2] * ratio_w), int(box[3] * ratio_h)]
        new_boxes.append(new_box)
    return new_boxes

def draw_box(answer, np_image):
    image = Image.fromarray(np.uint8(np_image)).convert('RGB')
    answer, box = extract_boxes(answer)
    image_w = image.width
    image_h = image.height

    bboxes = resize_bbox(box, image_w, image_h)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red")

    image.save("./ferret_bbox.jpg")
    return answer, bboxes, image

def get_ferret_response(prompt, img):
    responses = ask_ferret(prompt, img)
    for success, response in responses:
        if not success:
            print(response)

    answer = response[len(prompt):]
    final_answer, boxes, image_with_box = draw_box(answer, img)
    return final_answer, boxes, image_with_box