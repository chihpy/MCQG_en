"""
"""
import os

from src.feedback_lgc import FeedbackLgc
from utils import mkdir, json_load, json_dump

#####
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
#####

def get_feedback_in(qset):
    rv = {
        # qgen_init_in
        'clinical_note': qset['clinical_note'],
        'topic': qset['topic'],
        'keypoint': qset['keypoint'],
        # qgen_init_out
        'context': qset['context'],
        'question': qset['question'],
        'correct_answer': qset['correct_answer'],
        'distractor_options': qset['distractor_options'],  ###
        # rans_out
        'attempted_answer': qset['attempted_answer'],
        'reasoning': qset['reasoning'],
    }
    return rv

if __name__ == "__main__":
    # setup save_file_path
    DEST_DIR = os.path.join('data', 'outputs', 'test')
    mkdir(DEST_DIR)
    save_file_path = os.path.join(DEST_DIR, 'test_feedback.json')
    # input prepare
    input_file_path = os.path.join('data', 'inputs', 'inputs_feedback.json')
    assert os.path.isfile(input_file_path)
    fewshot_path = os.path.join('data', 'fewshot_source', 'feedback_fewshot.json')
    assert os.path.isfile(fewshot_path)
    rubrics_path = os.path.join('data', "input_source", "rubrics.json")
    assert os.path.isfile(rubrics_path)
    data = json_load(input_file_path)
    # initialization
    model = FeedbackLgc(fewshot_path, rubrics_path)
    # main loop
    rvs = []
    for idx, qset in enumerate(data):
        print("---"*5 + f"{idx + 1}" + "---"*5)
        if idx == 3:
            break
        query = get_feedback_in(qset)
        # invoke
#        context_response = model.feedback_component('context', query)
        response = model.feedback(query)
        # update
        qset.update({'feedback': response})
#        qset.update(response)

        rvs.append(qset)
    # dumps
    json_dump(save_file_path, rvs)