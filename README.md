# MCQG-SRefine.py_re-implement
Re-implementation of **"MCQG-SRefine"**
> **Disclaimer:** This is *not* the official implementation.  

This repository contains a Python-based re-implementation of the **MCQG-SRefine** model, originally proposed in the paper:
* \[arXiv] [**MCQG-SRefine: Multiple Choice Question Generation and Evaluation with Iterative Self-Critique, Correction, and Comparison Feedback**](https://arxiv.org/pdf/2410.13191)

The main objective of this project is to **explore LLM-based approaches** for automatic generation of **medical multiple-choice questions (MCQs)**.

This includes:

* **Question generation** – Creating high-quality medical MCQs using large language models.
* **Answer generation** – Producing correct answers along with detailed explanations.
* **Question evaluation** – Assessing correctness, difficulty, and discrimination indices.
* **Question refinement** – Iteratively improving questions based on evaluation feedback.

# Quick start
- `python oneshot.py`
    - data/outputs/oneshot_collector.json
- `python main.py`
    - data/outputs/qgen_auto_feedback.json

# Tasks
##  data prepare
- qbank vector database
    - `python build_qbank_db.py`
- inputs
    - data/fewshot_source
    - data/input_source
    - data/inputs

## qgen_init
- src/qgen_init_lgc.py
- `python src/qgen_init_lgc.py`
    - check prompt
- `python test_qgen_init.py`
    - input:
        - (clinical_note, topic, keypoint)
    - output: 
        - (context, question, correct_answer, distractor_options)
            - distractor_options is a list of options does not include correct answer

## reasoning_answer
- src/reasoning_answer_lgc.py
- `python test_reasoning_answer.py`
    - input:
        (context, question, options)
    - output: 
        (attempted_answer, reasoning)

## feedback
- src/feedback_lgc.py
- src/models/
    - context_feedback
    - correct_answer_feedback
    - distractor_options_feedback
    - question_feedback
    - reasoning_feedback
- `python test_feedback.py`
    - input:
        - (clinical_note, topic, keypoint)
        - (context, question, correct_answer, distractor_options)
    - output:
        - context_feedback
        - context_scroe
        - question_feedback
        - question_score
        - correct_answer_feedback
        - correct_answer_score
        - distractor_options_feedback
        - distractor_options_score
        - reasoning_feedback
        - reasoning_score
        - score
            - stop

## qgen_iterative
- src/qgen_iterative_lgc.py
- `python test_qgen_iterative.py`
    - input:
        - (clinical_note, topic, keypoint)
        - (context, question, correct_answer, distractor_options)
        - (attempted_answer, reasoning)
        - feedback
            - context_feedback
            - context_scroe
            - question_feedback
            - question_score
            - correct_answer_feedback
            - correct_answer_score
            - distractor_options_feedback
            - distractor_options_score
            - reasoning_feedback
            - reasoning_score

# Reference
- [MCQG-SRefine official github](https://github.com/bio-nlp/MedQG)
- [arxiv paper: MCQG-SRefine: Multiple Choice Question Generation and Evaluation
with Iterative Self-Critique, Correction, and Comparison Feedback](https://arxiv.org/pdf/2410.13191)
