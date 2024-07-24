import time

class Frame:
    def __init__(self, img, ID):
        self.img = img
        self.frameID = ID
        self.timestamp = time.time()

class Question:
    def __init__(self, text):
        self.text = text
        self.timestamp = time.time()

class Response:
    def __init__(self, text, question: Question):
        self.text = text
        self.question = question
        self.timestamp = time.time()
