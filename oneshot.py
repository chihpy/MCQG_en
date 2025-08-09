"""
"""
import os
import pandas as pd
from tqdm import tqdm

from utils import json_dump

from src.qgen_init_lgc import QgenInitLgc
from src.reasoning_answer_lgc import ReasoningAnswerLgc
from src.feedback_lgc import FeedbackLgc
from src.qgen_iterative_lgc import QgenIterativeLgc

def get_qgen_init_in(row):
    rv = {
        'clinical_note': row.clinical_note,
        'topic': row.topic,
        'keypoint': row.keypoint
    }
    return rv

def _get_option_txt(option_lst):
    txt = ''
    for opt in option_lst:
        txt += (opt + '\n')
    return txt

def _number_to_letter(n):
    return chr(ord('A') + n)

def _make_distractor_dict(distractor_options):
    rv = dict()
    for idx, opt in enumerate(distractor_options):
        rv[f"{_number_to_letter(idx)}"] = opt
    return rv

def get_rans_in(qgen_init_out):
    option_lst = qgen_init_out['distractor']['options']
    option_txt = _get_option_txt(option_lst)
    rans_in = {
        'context': qgen_init_out['context'],
        'question': qgen_init_out['question'],
        'options': option_txt
    }
    return rans_in

def get_feedback_in(qgen_init_in, qgen_init_out, rans_out):
    feedback_in = {
        'clinical_note': qgen_init_in['clinical_note'],
        'topic': qgen_init_in['topic'],
        'keypoint': qgen_init_in['keypoint'],

        'context': qgen_init_out['context'],
        'question': qgen_init_out['question'],
        'correct_answer': qgen_init_out['correct_answer'],
        'distractor_options': _make_distractor_dict(qgen_init_out['distractor_options']),  ###

        'attempted_answer': rans_out['attempted_answer'],
        'reasoning': rans_out['reasoning']
    }
    return feedback_in

def get_qgen_it_in(qgen_init_in, qgen_init_out, rans_out, feedback_out):
    qgen_it_in = {
        # qgen_init_in
        'clinical_note': qgen_init_in['clinical_note'],
        'topic': qgen_init_in['topic'],
        'keypoint': qgen_init_in['keypoint'],
        # qgen_init_out
        'context': qgen_init_out['context'],
        'question': qgen_init_out['question'],
        'correct_answer': qgen_init_out['correct_answer'],
        'distractor_options': _make_distractor_dict(qgen_init_out['distractor_options']),  ###
        
        'attempted_answer': rans_out['attempted_answer'],
        'reasoning': rans_out['reasoning'],
        # feedback_out
        'feedback': feedback_out
    }
    return qgen_it_in

if __name__ == "__main__":
    input_file_path = os.path.join('data', 'input_source', 'inputs_qgen_auto_feedback.jsonl')
    df = pd.read_json(input_file_path, orient="records", lines=True)
    print(f"df.columns: {df.columns}")

    db_dir = "data/qbank_embedding/faiss/"
    qgen_init = QgenInitLgc(fewshot_path='', db_dir=db_dir)

    answer_fewshot_path = os.path.join('data', 'fewshot_source', 'reasoning_answer_fewshot.json')
    ranswer = ReasoningAnswerLgc(answer_fewshot_path)

    feedback_fewshot_path = os.path.join('data', 'fewshot_source', 'feedback_fewshot.json')
    rubrics_path = os.path.join('data', "input_source", "rubrics.json")
    feedback = FeedbackLgc(feedback_fewshot_path, rubrics_path)

    iterative_fewshot_path = os.path.join('data', 'fewshot_source', 'iterative_fewshot.json')
    qgen_it = QgenIterativeLgc(iterative_fewshot_path)

    collector = {}
    ### qgen init in: begin
#    for idx, row in tqdm(df.iterrows(), total=len(df)):
    for idx, row in df.iterrows():
        qgen_init_in = get_qgen_init_in(row)
        if idx == 0:
            break

    collector['qgen_init_in'] = qgen_init_in
    ### qgen init in: end

    ### qgen init out: begin
    qgen_init_out = qgen_init.qgen(qgen_init_in)
    collector['qgen_init_out'] = qgen_init_out
    ### qgen init out: end

    ### rans in: begin
    rans_in = get_rans_in(qgen_init_out)
    collector['rans_in'] = rans_in
    ### rans in: end

    ### rans out: begin
    rans_out = ranswer.reasoning_answer(rans_in)
    collector['rans_out'] = rans_out
    ### rans out: end

    ### feedback_in: begin
    feedback_in = get_feedback_in(qgen_init_in, qgen_init_out, rans_out)
    collector['feedback_in'] = feedback_in
    ### feedback_in: end

    ### feedback_out: begin
    feedback_out = feedback.feedback(feedback_in)
    collector['feedback_out'] = feedback_out
    ### feedback_out: end

    ### qgen_it in: begin
    qgen_it_in = get_qgen_it_in(qgen_init_in, qgen_init_out, rans_out, feedback_out)
    collector['qgen_it_in'] = qgen_it_in
    ### qgen_it in: end

    ### qgen_it out: begin
    qgen_it_out = qgen_it.qgen_iterative(qgen_it_in)
    collector['qgen_it_out'] = qgen_it_out
    ### qgen_it out: end

    ### qgen_ans_in: begin
    rans2_in = get_rans_in(qgen_it_out['qgen_iterative'])
    collector['rans2_in'] = rans2_in
    ### qgen_ans_in: end

    ### qgen_ans_out: begin
    rans_out = ranswer.reasoning_answer(rans2_in)
    collector['rans2_out'] = rans_out
    ### qgen_ans_out: end
    json_dump('data/outputs/oneshot_collector.json', collector)