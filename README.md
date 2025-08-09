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

# Tasks
0. data prepare
- qgen_init
    - qbank vector database
        - `python build_qbank_db.py`
    - inputs

1. qgen_init
- src/qgen_init_lgc.py
- `python src/qgen_init_lgc.py`
- `python test_qgen_init.py`
- input:
    - (clinical_note, topic, keypoint)
- output: 
    - (context, question, correct_answer, distractor_options)
        - distractor_options is a list of options does not include correct answer

# Reference
- [MCQG-SRefine official github](https://github.com/bio-nlp/MedQG)
- [arxiv paper: MCQG-SRefine: Multiple Choice Question Generation and Evaluation
with Iterative Self-Critique, Correction, and Comparison Feedback](https://arxiv.org/pdf/2410.13191)
