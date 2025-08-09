"""
"""
import os
from tqdm import tqdm
import pandas as pd

from src.qgen_init_lgc import QgenInitLgc
from src.reasoning_answer_lgc import ReasoningAnswerLgc
from src.feedback_lgc import FeedbackLgc
from src.qgen_iterative_lgc import QgenIterativeLgc

from utils import mkdir, json_dump

def check_stop(n_attempts, max_attempts):
    if n_attempts >= max_attempts:
        return False
    else:
        return True

def dict2str(content_dict):
    content = ''
    for key in content_dict.keys():
        content += '\n' + key + ': ' + content_dict[key]
    return content

def lst2str(dist_lst):
    text = ''
    for op in dist_lst:
        text+=f"{op}\n"
    return text

def get_feedback_in(rans, query, qset):
    """
    query: {clinical_note, topic, keypoint}
    qgen: {context, question, correct_answer, distractor_options}
    qans: {attempted_answer, reasoning}
    qfeedback
    - [ ] 確定一下distractor到底有沒有answer
    """
    rv = {
        'clinical_note': query['clinical_note'],
        'topic': query['topic'],
        'keypoint': query['keypoint'],
        'context': qset['context'],
        'question': qset['question'],
        'correct_answer': qset['correct_answer'],
        'distractor_options': qset['distractor_options'],
        'attempted_answer': rans['attempted_answer'],
        'reasoning': rans['reasoning'],
    }
    return rv

def get_qgen_init_in(clinical_note, topic, keypoint):
    rv = {
        'clinical_note': clinical_note,
        'topic': topic,
        'keypoint': keypoint
    }
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

def _get_option_txt(option_lst):
    txt = ''
    for opt in option_lst:
        txt += (opt + '\n')
    return txt

def get_feedback_in(qgen_init_in, qgen_out, rans_out):
    feedback_in = {
        'clinical_note': qgen_init_in['clinical_note'],
        'topic': qgen_init_in['topic'],
        'keypoint': qgen_init_in['keypoint'],

        'context': qgen_out['context'],
        'question': qgen_out['question'],
        'correct_answer': qgen_out['correct_answer'],
        'distractor_options': _make_distractor_dict(qgen_out['distractor_options']),  ###

        'attempted_answer': rans_out['attempted_answer'],
        'reasoning': rans_out['reasoning']
    }
    return feedback_in

def _make_distractor_dict(distractor_options):
    rv = dict()
    for idx, opt in enumerate(distractor_options):
        rv[f"{_number_to_letter(idx)}"] = opt
    return rv

def _number_to_letter(n):
    return chr(ord('A') + n)

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

def qgen_auto_feedback(clinical_note, topic, keypoint, max_attempts=3):
    """
    """
    db_dir = "data/qbank_embedding/faiss/"
    #########################################################
    # initialization
    qgen_init = QgenInitLgc(fewshot_path='', db_dir=db_dir)
    ## reasoning_answer_lgc
    answer_fewshot_path = os.path.join('data', 'fewshot_source', 'reasoning_answer_fewshot.json')
    ranswer = ReasoningAnswerLgc(answer_fewshot_path)
    ## feedback_lgc
    feedback_fewshot_path = os.path.join('data', 'fewshot_source', 'feedback_fewshot.json')
    rubrics_path = os.path.join('data', "input_source", "rubrics.json")
    feedback = FeedbackLgc(feedback_fewshot_path, rubrics_path)
    ## qgen_iterate_lgc
    iterative_fewshot_path = os.path.join('data', 'fewshot_source', 'iterative_fewshot.json')
    qgen_it = QgenIterativeLgc(iterative_fewshot_path)
    #########################################################
    n_attempts = 0
    qgen_init_in = get_qgen_init_in(clinical_note, topic, keypoint)
    # main loop
    rvs = []
    rvs.append({'qgen_in': qgen_init_in})
    while check_stop(n_attempts, max_attempts):
        if n_attempts == 0:
            qgen_out = qgen_init.qgen(qgen_init_in)  # context, question, correct_answer, distractor_options, distractor
            rans_in = get_rans_in(qgen_out)
            rans_out = ranswer.reasoning_answer(rans_in)  # attempted_answer, reasoning
        else:
            qgen_it_in = get_qgen_it_in(qgen_init_in, qgen_out, rans_out, feedback_out)
            qgen_it_out = qgen_it.qgen_iterative(qgen_it_in)
            qgen_out = qgen_it_out['qgen_iterative']
            rans_in = get_rans_in(qgen_out)
            rans_out = ranswer.reasoning_answer(rans_in)  # attempted_answer, reasoning

        rvs.append({'qgen_out': qgen_out})
        rvs.append({'rans_out': rans_out})
        feedback_in = get_feedback_in(qgen_init_in, qgen_out, rans_out)
        feedback_out = feedback.feedback(feedback_in)
        # append result
        print("# update")
        rvs.append({'feedback_out': feedback_out})
        if feedback_out['score']['stop']:
            break
        n_attempts+=1
    return rvs

def make_rv(row, response):
    rv = {
        "clinical_note": row['clinical_note'],
        "topic": row['topic'],
        "keypoint": row['keypoint'],
        "response": response,
    }
    return rv

def main(input_file_path, output_dir):
    df = pd.read_json(input_file_path, orient="records", lines=True)
    rvs = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ##########
        if idx == 3:
            break
        ##########
        response = qgen_auto_feedback(clinical_note=row['clinical_note'], 
                                      topic=row['topic'], 
                                      keypoint=row['keypoint'],
                                      )
        rv = make_rv(row, response)
        rvs.append(rv)
    # dump    
    output_file_path = os.path.join(output_dir, 'qgen_auto_feedback.json')
    json_dump(output_file_path, rvs)


if __name__ == "__main__":
    input_file_path = os.path.join('data', 'input_source', 'inputs_qgen_auto_feedback.jsonl')
    output_dir = os.path.join('data', 'outputs')
    mkdir(output_dir)
    main(input_file_path, output_dir)

