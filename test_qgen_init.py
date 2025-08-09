"""
"""
import os
import pandas as pd
from tqdm import tqdm

from src.qgen_init_lgc import QgenInitLgc
from utils import json_load, json_dump, mkdir
#####
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
#####
def make_query(row):
    query = {}
    query['clinical_note'] = row.clinical_note
    query['topic'] = row.topic
    query['keypoint'] = row.keypoint
    return query

def make_rv(row, response):
    response.update({
        'clinical_note': row.clinical_note,
        'topic': row.topic,
        'keypoint': row.keypoint,
    })
    return response

if __name__ == "__main__":
    # setup save_file_path
    DEST_DIR = os.path.join('data', 'outputs', 'test')
    mkdir(DEST_DIR)
    save_file_path = os.path.join(DEST_DIR, 'test_qgen_init.json')
    # input prepare
    input_file_path = os.path.join('data', 'input_source', 'inputs_qgen_auto_feedback.jsonl')
    assert os.path.isfile(input_file_path)
    fewshot_path = os.path.join('data', 'fewshot_source', 'qgen_init_fewshot.json')
    assert os.path.isfile(fewshot_path)
    df = pd.read_json(input_file_path, orient="records", lines=True)
    # initialization
    db_dir = "data/qbank_embedding/faiss/"
    model = QgenInitLgc(fewshot_path='', db_dir=db_dir)
    rvs = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ##########
        if idx == 2:
            break
        ##########
        query = make_query(row)
        response = model.qgen(query)
        rv = make_rv(row, response)
        rvs.append(rv)
    # dump
    
    json_dump(save_file_path, rvs)