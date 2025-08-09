"""
"""
import re
import os
import random
random.seed(42)
import json

class QgenIterativePrompt():
    def __init__(self, fewshot_path):
        self.example_prompt = self.get_iterate_example(fewshot_path)
    
    def get_prompt(self, query):
        query_prompt = self.get_query_prompt(query)
        return self.example_prompt + query_prompt

    def get_query_prompt(self, query):
        query_template = self.get_query_template()
        clinical_note = query['clinical_note']
        topic = query['topic']
        keypoint = query['keypoint']
        feedback = query['feedback']
        example_text = query_template.format(
                    context=query["context"],
                    question=query["question"],

                    attempted_answer=query['attempted_answer'],
                    reasoning=query['reasoning'],

                    correct_answer=query["correct_answer"],
                    distractor_options=query["distractor_options"], ###

                    context_feedback=self._dict2str(feedback["context_feedback"]),
                    context_score=feedback["context_score"],
                    question_feedback=self._dict2str(feedback["question_feedback"]),
                    question_score=feedback["question_score"],
                    correct_answer_feedback=self._dict2str(feedback["correct_answer_feedback"]),
                    correct_answer_score=feedback["correct_answer_score"],
                    distractor_option_feedback=self._dict2str(feedback["distractor_options_feedback"]),
                    distractor_option_score=feedback["distractor_options_score"],
                    reasoning_feedback=self._dict2str(feedback['reasoning_feedback']),
                    reasoning_score=feedback['reasoning_score']
                )
        
        prefix = self.get_prefix()
        query_prompt = prefix.format(clinical_note=clinical_note,
                                     topic=topic,
                                     keypoint=keypoint)
        query_prompt+=example_text
        suffix = self.get_suffix()
        return query_prompt + "\n\n" + suffix
    
    def get_iterate_example(self, fewshot_path):
        """
        cnote, topic, keypoint
        (example) imp ver (example)
        """
        infix = self.get_infix()
        example_template = self.get_example_template()
        fewshot = self._json_load(fewshot_path)
        examples = []
        clinical_note = fewshot['clinical_note']
        topic = fewshot['topic']
        keypoint = fewshot['keypoint']
        content_to_feedback = fewshot['content_to_feedback']
        for example in content_to_feedback:
            example_text = example_template.format(
                    context=example["context"],
                    question=example["question"],
                    correct_answer=example["correct_answer"],
                    distractor_options = example["distractor_options"],  ###
                    context_feedback=self._dict2str(example["context_feedback"]),
                    context_score = example["context_score"],
                    question_feedback=self._dict2str(example["question_feedback"]),
                    question_score = example["question_score"],
                    correct_answer_feedback=self._dict2str(example["correct_answer_feedback"]),
                    correct_answer_score = example["correct_answer_score"],
                    distractor_option_feedback = self._dict2str(example["distractor_option_feedback"]),  ###
                    distractor_option_score = example["distractor_option_score"]  ###
                )
            examples.append(example_text)
        prefix = self.get_prefix()
        example_prompt = prefix.format(clinical_note=clinical_note,
                                       topic=topic,
                                       keypoint=keypoint) 
        example_prompt+= infix.join(examples)
        return example_prompt
    
    def get_prefix(self):
        header = (
            "Clinical Note: {clinical_note}\n"
            "Topic: {topic}\n"
            "Keypoint: {keypoint}\n"
        )
        return header
    
    def get_infix(self):
        infix = "\nImproved version of the above components using their respective feedbacks: \n\n"
        return infix

    def get_suffix(self):
        suffix = (
            "Improve the context,question, correct answer and distractor options "
            "using each previous components' feedback and the reasoning feedback.\n"
            "Generate a context, question, correct answer and distractor options "
            "that can achieve high scores on all the above feedback rubrics, "
            "given the clinical note, keypoint and topic. "
            "Do not generate the feedback for any of the component."
        )
        return suffix
    
    def get_example_template(self):
        example_template = (
            "Context: {context}\n"
            "Question: {question}\n"
            "Correct answer: {correct_answer}\n"
            "Distractor Options: {distractor_options}\n"
            "Feedback for the above components:\n"
            "Context feedback: {context_feedback}\n"
            "Context score: {context_score}\n"
            "Question feedback: {question_feedback}\n"
            "Question score: {question_score}\n"
            "Correct answer feedback: {correct_answer_feedback}\n"
            "Correct answer score: {correct_answer_score}\n"
            "Distractor options feedback: {distractor_option_feedback}\n"
            "Distractor options score: {distractor_option_score}\n"
        )
        return example_template
    
    def get_query_template(self):
        # 這個主要是因為他的example裡面沒有 reasoning
        query_template = (
            "Context: {context}\n"
            "Question: {question}\n"
            ###
            "Attempted answer: {attempted_answer}\n"
            "Reasoning: {reasoning}\n"
            ###
            "Correct answer: {correct_answer}\n"
            "Distractor Options: {distractor_options}\n"
            "Feedback on the generated content with respect to various rubrics.\n"
            "Context feedback: {context_feedback}\n"
            "Context score: {context_score}\n"
            "Question feedback: {question_feedback}\n"
            "Question score: {question_score}\n"
            "Correct answer feedback: {correct_answer_feedback}\n"
            "Correct answer score: {correct_answer_score}\n"
            "Distractor options feedback: {distractor_option_feedback}\n"
            "Distractor options score: {distractor_option_score}\n"
            ###
            "Reasoning feedback: {reasoning_feedback}\n"
            "Reasoning score: {reasoning_score}"
            ###
        )
        return query_template

    def _json_load(self, file_path):
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        return data_dict

    def _dict2str(self, content_dict):
        content = ''
        for key in content_dict.keys():
            content += '\n' + key + ': ' + content_dict[key]
        return content

from langchain_openai import ChatOpenAI

class QgenIterativeLgc():
    def __init__(self, fewshot_path):
        self.iterative_prompt_generator = QgenIterativePrompt(fewshot_path)
        self._llm_init()

    def _llm_init(self):
        model_name='gpt-4.1-mini'
        temperature=0.0
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def qgen_iterative(self, query):
        prompt = self.iterative_prompt_generator.get_prompt(query)
        response = self.llm.invoke(prompt)
        rv = self._output_parser(response)
        return rv

    def _output_parser(self, response):
        response_row = response.content
        rv = {}
        rv[f'qgen_iterative_row'] = response_row

        parts = response_row.split('\n\n')
        for part in parts:
            if part.strip().lower().startswith('context:'):
                rv['context'] = part.split(':', 1)[1].strip()
            elif part.strip().lower().startswith('question:'):
                rv['question'] = part.split(':', 1)[1].strip()
            elif part.strip().lower().startswith('correct answer:'):
                rv['correct_answer'] = part.split(':', 1)[1].strip()
            elif part.strip().lower().startswith('distractor options:'):
                rv['distractor_options'] = part.split(':', 1)[1].strip()
        # distractor parser
        distractor_lst = self.str2lst(rv['distractor_options'])
        rv['distractor_options'] = distractor_lst
        # options_gen
        options, answer_index = self.options_gen(distractor_lst, rv['correct_answer'])
        rv['distractor'] = {
            'answer_index': answer_index,
            'options': options,
            'distractor_lst': distractor_lst
        }
        return {'qgen_iterative': rv}

    def options_gen(self, distractor_lst, answer):
        options = distractor_lst + [answer]
        random.shuffle(options)
        answer_index = self.number_to_letter(options.index(answer))
        rvs = []
        for idx, option in enumerate(options):
            rvs.append(f"{self.number_to_letter(idx)}: {option}")
        return rvs, answer_index

    def str2lst(self, distractor):
        items = distractor.split('\n')
        items = [re.sub(r'^[a-z]\)\s*', '', item.strip()) for item in items]
        return items
    
    def number_to_letter(self, n):
        return chr(ord('A') + n)

def json_load(file_path):
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    return data_dict

if __name__ == "__main__":
    SOURCE_DIR = "/home/poyuan/workspace/MCQG/MCQG/data"
    fewshot_path = os.path.join(SOURCE_DIR, "fewshot_source", 'iterative_fewshot.json')
    assert os.path.isfile(fewshot_path)
    input_file_path = os.path.join(SOURCE_DIR, 'input_source', 'inputs_iterative.json')
    qgen_iterative_prompt_generator = QgenIterativePrompt(fewshot_path)
    data = json_load(input_file_path)
    qset = data[0]
    prompt = qgen_iterative_prompt_generator.get_prompt(qset)
    print(prompt)
