import json
import re
import math
import pre_process as pp


def load_json(json_file):
    with open(json_file) as json_data:
        return json.load(json_data)


def get_questions(wiki):
    questions = []
    for qa in wiki['qa']:
        questions.append(pp.process_question(qa['question'], False))
    return questions


def isnumber(num):
    regex = re.compile(r"^(-?\d+)((\.|-|:|,)?\d*)?$")
    if re.match(regex, num):
        return True
    else:
        return False


def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    else:
        return 1 / (1 + math.exp(-x))

