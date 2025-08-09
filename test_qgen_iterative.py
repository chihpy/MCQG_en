"""
"""
import os
from src.qgen_iterative_lgc import QgenIterativeLgc
from utils import mkdir, json_load, json_dump
#####
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
#####

def get_iterative_in(qset):
    """deprecated
    """
    #print(qset.keys())
    rv = {}
    rv['clinical_note'] = qset['clinical_note']
    rv['keypoint'] = qset['keypoint']
    rv['topic'] = qset['topic']
    rv['feedback'] = qset['feedback']  # content_to_fb
    return rv

if __name__ == "__main__":
    # setup save_file_path
    DEST_DIR = os.path.join('data', 'outputs', 'test')
    mkdir(DEST_DIR)
    save_file_path = os.path.join(DEST_DIR, 'test_qgen_iterative.json')
    # input prepare
    # TODO: input, not input_source
    input_file_path = os.path.join('data', 'inputs', 'inputs_iterative.json')
    assert os.path.isfile(input_file_path)
    fewshot_path = os.path.join('data', 'fewshot_source', 'iterative_fewshot.json')
    assert os.path.isfile(fewshot_path)
    data = json_load(input_file_path)
    # initialization
    model = QgenIterativeLgc(fewshot_path)
    # main loop
    rvs = []
    for idx, qset in enumerate(data):
        print("---"*5 + f"{idx + 1}" + "---"*5)
        if idx == 3:
            break
        # invoke
        response = model.qgen_iterative(qset)
        # update 
        qset.update(response)
        rvs.append(qset)
    # dumps
    json_dump(save_file_path, rvs)