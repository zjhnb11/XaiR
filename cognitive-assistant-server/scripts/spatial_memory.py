import json
from .model_interface.ferret_with_gpt import ask_gpt, ask_gpt_3_5

class HistoryLogger:
    def __init__(self, data_collector):
        self.data_collector = data_collector

        self.history = dict()
        self.session_started = False
        self.sessionID = 0
        self.history[self.sessionID] = dict()

    def start(self, sessionID, frameID):
        if not self.session_started:
            self.session_started = True
            self.startFrame = frameID
            self.sessionID = sessionID
            self.endFrameID = None
            self.history[sessionID] = dict()

    def stop(self, frameID):
        self.endFrameID = frameID

    def session_finish_processing(self):
        self.session_started = False
        self.print_history()

    def add_scene_description(self, time, poseID, description):
        if self.session_started:
            if time not in self.history[self.sessionID].keys():
                self.history[self.sessionID][time] = {"poseID":poseID,
                                    "sceneDescription": description
                                    }
            else:
                self.history[self.sessionID][time]["sceneDescription"] = description

    def add_question_asked(self, time, poseID, question, answer):
            if time not in self.history[self.sessionID].keys():
                self.history[self.sessionID][time] = {"poseID":poseID,
                                    "questionAsked": {
                                        "question": question,
                                        "answer":answer
                                    }
                                }
            else:
                self.history[self.sessionID][time]["questionAsked"] = {
                                        "question": question,
                                        "answer":answer
                                        }

    def print_history(self):
        j = json.dumps(self.history, indent=4, sort_keys=True)
        with open("sample_%d.json" % rep, "w") as outfile:
            outfile.write(j)
        self.get_summary_gpt(self.sessionID)

    def get_summary_gpt(self, sessionID):
        scene_desc = ""
        for time in self.history[sessionID].keys():
            if "sceneDescription" in self.history[sessionID][time].keys():
                scene_desc += self.history[sessionID][time]["sceneDescription"] + '/n'
        print(scene_desc)
        prompt = 'I took a video of me setting up the plate and cutlery for a meal. I gave an AI model the video in several' +\
                 'segments and asked it to describe what I was doing. It is not the best model' +\
                 ' but it got the general idea. Here are the descriptions of each video segment it' +\
                 ' came up with. Based on this, can you give me a concise and practical list of' +\
                 ' instructions to do what I am doing in the video? Each line is a desription of each' +\
                 ' segment. Answer based on what i did in the video only and not your own knowledge. Do not add any extra steps that has not been done in the video: \n' + scene_desc
        answer = ask_gpt_3_5(prompt)
        print(answer)
        with open("overall_inst_%d.txt" % rep, "w") as outfile:
            outfile.write(answer)
