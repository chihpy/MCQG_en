"""
"""
import os

from src.reasoning_answer_lgc import ReasoningAnswerLgc
from utils import mkdir, json_load, json_dump
#####
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
#####

def get_reasoning_answer_in(qset):
    context = '.'.join(qset['question'].split('.')[:-1])
    question = qset['question'].split('.')[-1]
    option_txt = ' '.join([f"{k} :" + f"{v}" for k,v in qset['options'].items()])
    rv = {
        "context": context,
        "question": question,
        "options": option_txt
    }
    return rv

if __name__ == "__main__":
    # setup save_file_path
    DEST_DIR = os.path.join('data', 'outputs', 'test')
    mkdir(DEST_DIR)
    save_file_path = os.path.join(DEST_DIR, 'test_reasoning_answer.json')
    # input prepare
    input_file_path = os.path.join('data', 'input_source', 'inputs_reasoning_answer.json')
    assert os.path.isfile(input_file_path)
    fewshot_path = os.path.join('data', 'fewshot_source', 'reasoning_answer_fewshot.json')
    assert os.path.isfile(fewshot_path)
    data = json_load(input_file_path)
    # initialization
    model = ReasoningAnswerLgc(fewshot_path)
    # main loop
    rvs = []
    for idx, qset in enumerate(data):
        print("---"*5 + f"{idx + 1}" + "---"*5)
        if idx == 3:
            break
        query = get_reasoning_answer_in(qset)
        #print(query)
        # invoke
        response = model.reasoning_answer(query)
        # update 
        qset.update(response)
        rvs.append(qset)
    # dumps
    json_dump(save_file_path, rvs)