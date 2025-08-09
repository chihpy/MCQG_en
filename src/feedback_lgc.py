"""
1. fewshot example formatt
2. example prompt
3. suffix
4. instructions
5. get prompt
"""
import re
import os
import json

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser

import sys
sys.path.append("/home/poyuan/workspace/MCQG/MCQG/src")
from models.context_feedback import context_feedback
from models.correct_answer_feedback import correct_answer_feedback
from models.distractor_options_feedback import distractor_options_feedback
from models.question_feedback import question_feedback
from models.reasoning_feedback import reasoning_feedback

class FeedbackPrompt():
    def __init__(self, fewshot_path, rubrics_path):
        self.fewshot_path = fewshot_path
        self.rubrics_path = rubrics_path
        self._set_feedback_dict()
    
    def get_rubrics(self, component_name):
        rubrics = self._json_load(self.rubrics_path)
        rub_comp_feedback = {k:rubrics[k] for k in rubrics.keys() if k.lower().find(component_name) != -1}
        rub_comp_feedback[list(rub_comp_feedback.keys())[0]] = self._dict2str(rub_comp_feedback[list(rub_comp_feedback.keys())[0]])
        return rub_comp_feedback

    def get_prompt(self, qset, component_name):
        reasoning_rubrics = self.get_rubrics(component_name)
        qset['component_name'] = component_name
        qset['reasoning_rubrics'] = reasoning_rubrics
        fewshot_prompt = self._get_fewshot_prompt(component_name)
        return fewshot_prompt.invoke(qset)

    def get_reasoning_prompt(self, qset, component_name='reasoning'):
        reasoning_rubrics = self.get_rubrics(component_name)
        qset['component_name'] = component_name
        qset['reasoning_rubrics'] = reasoning_rubrics
        suffix = self._get_reasoning_suffix()
        instructions = self._get_instructions(component_name)
        reasoning_prompt = PromptTemplate(
            template=suffix,
            input_variables=["component_name",
                             "clinical_note", 
                             "topic",
                             "keypoint",
                             "context",
                             "question",
                             "correct_answer",
                             "attempted_answer",
                             "reasoning",
                             "distractor_options",
                             "reasoning_rubrics"],
            partial_variables={"format_instructions": instructions})
        #fewshot_prompt = self._get_fewshot_prompt(component_name)
        return reasoning_prompt.invoke(qset)
    
    def _set_feedback_dict(self):
        self.feedback_dict = {
            "context":PydanticOutputParser(pydantic_object=context_feedback),
            "question":PydanticOutputParser(pydantic_object=question_feedback),
            "correct_answer":PydanticOutputParser(pydantic_object=correct_answer_feedback),
            "distractor_options":PydanticOutputParser(pydantic_object=distractor_options_feedback),
            "reasoning":PydanticOutputParser(pydantic_object=reasoning_feedback)
        }
    def _get_instructions(self, comp_name):
        parser = self.feedback_dict[comp_name]
        return parser.get_format_instructions()
    
    def _dict2str(self, content_dict):
        content = ''
        for key in content_dict.keys():
            content += '\n' + key + ': ' + content_dict[key]
        return content
    
    def _get_fewshot_example(self, comp_name):
        """
        - example = self._get_fewshot_example('context')
        """
        dictt = self._json_load(self.fewshot_path)
        # 'cname', 'cname_feedback', 'cname_score'
        ex_comp_fb = {k:dictt[k] for k in dictt.keys() if k.lower().find(comp_name) != -1}
        # 'cnote', 'question', 'answer', 'distractor_options'
        ex_content = {k:dictt[k] for k in dictt.keys() if (k.lower().find('feedback') == -1 and k.lower().find('score') == -1)}
        # key('cname_feedback)
        feedback_key = [j for j in ex_comp_fb.keys() if j.lower().find('feedback') != -1][0]
        # key('cname_score')
        score_key = [j for j in ex_comp_fb.keys() if j.lower().find('score') != -1][0]
        # add string formatted feedback
        ex_comp_fb['feedback'] = self._dict2str(ex_comp_fb[feedback_key])
        # add score
        ex_comp_fb['score'] = ex_comp_fb[score_key]
        # add cname
        ex_comp_fb['component_name'] = comp_name
        # remove 'cname_score'
        del ex_comp_fb[score_key]
        # remove 'cname_feedback'
        del ex_comp_fb[feedback_key]
        # add example content
        ex_comp_fb.update(ex_content)
        # list-lization
        ex_comp = [ex_comp_fb]
        return ex_comp

    def _get_example_template(self):
        template = (
            "Clinical note: {clinical_note}\n"
            "Topic: {topic}\n"
            "Keypoint: {keypoint}\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Correct answer: {correct_answer}\n"
            "Distractor options: {distractor_options}\n"
            "{component_name} feedback: {feedback}\n"
            "{component_name} score: {score}"
        )
        example_template = PromptTemplate(
            input_variables=["clinical_note", "topic", "keypoint", "context", "question", "correct_answer", "distractor_options", "component_name", "feedback", "score"],
            template=template
            )
        return example_template

    def _get_suffix(self):
        suffix = (
            "In addition to the scoring rubrics in the examples above, "
            "give feedback and score the {component_name} using the attempted answer's(correct/incorrect) "
            "reasoning-based rubrics and their definitions below.\n"
            "Please include both the previous scoring rubrics and the following reasoning-based rubrics "
            "before giving the feedback for a particular aspect and "
            "add up the scores for all the aspects for the total scores of the {component_name}.\n"
            "Many of these feedback points for the {component_name} depend upon "
            "the reasoning and the attempted answer correctness so consider that while giving feedback for the {component_name}.\n"
            "{component_name} reasoning-based rubrics: {reasoning_rubrics}\n"
            "Give the output in just this format: {format_instructions}\n"
            "Output just the JSON instance and nothing else.\n"
            "Clinical note: {clinical_note}\n"
            "Keypoint: {keypoint}\n"
            "Topic: {topic}\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Correct answer: {correct_answer}\n"
            "Attempted answer: {attempted_answer}\n"
            "Reasoning: {reasoning}\n"
            "Distractor options: {distractor_options}\n"
        )
        return suffix

    def _get_reasoning_suffix(self):
        suffix = (
            'Give supporting textual feedback for each aspect and score(out of 5 for each aspect, in the format "2/5"'
            "if the score for that aspect is 2, also give supporting evidence for that score) the {component_name} using the attempted answer's(correct/incorrect) reasoning-based rubrics and their definitions below. "
            "Please include the following reasoning-based rubrics before giving the feedback for a particular aspect and add up the scores for all the aspects for the total score of the {component_name}. "
            "Many of these feedback points for the {component_name} depend upon the reasoning and the attempted answer correctness so consider that while giving feedback for the {component_name}. "
            "{component_name} rubrics: {reasoning_rubrics} \n"
            "Give the output in just this format: {format_instructions} "
            "Output just the JSON instance and nothing else. "
            "Clinical note: {clinical_note} \n"
            "Keypoint: {keypoint}\n"
            "Topic: {topic}\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Correct answer: {correct_answer}\n"
            "Attempted answer: {attempted_answer}\n"
            "Reasoning: {reasoning}\n"
            "Distractor options: {distractor_options}\n"
        )
        return suffix
 
    def _get_fewshot_prompt(self, component_name):
        example = self._get_fewshot_example(component_name)
        example_template = self._get_example_template()
        suffix = self._get_suffix()
        instructions = self._get_instructions(component_name)

        feedback_fewshot_prompt = FewShotPromptTemplate(
            examples=example,
            example_prompt=example_template,
            suffix=suffix,
            input_variables=["component_name", 
                             "clinical_note", 
                             "topic",
                             "keypoint",
                             "context",
                             "question",
                             "correct_answer",
                             "attempted_answer",
                             "reasoning",
                             "distractor_options",
                             "reasoning_rubrics"],
            partial_variables={"format_instructions": instructions})
        return feedback_fewshot_prompt
    
    def _json_load(self, file_path):
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        return data_dict

from langchain_openai import ChatOpenAI

class FeedbackLgc():
    def __init__(self, fewshot_path, rubrics_path):
        self.feedback_prompt_generator = FeedbackPrompt(fewshot_path, rubrics_path)
        self._llm_init()

    def _llm_init(self):
        model_name='gpt-4.1'
        temperature=0.0
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    def feedback(self, query):
        components = ['context', 'question', 'correct_answer', 'distractor_options']
        feedback = {}
        for component in components:
            print(f"- {component} feedback ...")
            feedback.update(self.feedback_component(component, query))
        component = 'reasoning'
        print(f"- {component} feedback ...")
        feedback.update(self.feedback_reasoning(component, query))

        score = self.score_computer(feedback)
        feedback['score'] = score
        return feedback

    def feedback_component(self, component_name, query):
        # get prompt
        prompt = self.feedback_prompt_generator.get_prompt(query, component_name)
        response = self.llm.invoke(prompt)
        rv = self._output_parser(component_name, response)
        return rv

    def feedback_reasoning(self, component_name, query):
        # get prompt
        prompt = self.feedback_prompt_generator.get_reasoning_prompt(query, component_name)
        response = self.llm.invoke(prompt)
        rv = self._output_parser(component_name, response)
        return rv

    def _output_parser(self, component_name, response):
        print(f"  {component_name} parser")
        response_row = response.content
        rv = {}
        rv[f'{component_name}_response_row'] = response_row
        try:
            json_str = response_row.strip("`json").strip("`")
            parsed = json.loads(json_str)
        except Exception as e:
            print("parser err ", e)
            return rv
        parsed.update(rv)
        return parsed
    
    def score_computer(self, feedback):
        # get score keys
        score_keys = [key for key in feedback.keys() if 'score' in key]
        rv = {}
        stop = True
        for key in score_keys:
            name = key.removesuffix('_score')
            score = self.get_dec_score(feedback[key])
            rv[name] = score
            if stop:
                if score < 0.9:
                    stop = False
        rv['stop'] = stop
        return rv

    def get_dec_score(self, score):
        split = score.split('/')
        num = int(re.sub("[^0-9]", "", split[0]))
        deno = int(re.sub("[^0-9]", "", split[1]))
        return num/deno

def get_feedback_in(qset):
    rv = {
        'clinical_note': qset['clinical_note'],
        'topic': qset['topic'],
        'keypoint': qset['keypoint'],
        'context': qset['context'],
        'question': qset['question'],
        'correct_answer': qset['correct_answer'],
        'distractor_options': qset['distractor_options'],
        'attempted_answer': qset['attempted_answer'],
        'reasoning': qset['reasoning'],
    }
    return rv

def json_load(file_path):
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    return data_dict

if __name__ == "__main__":
    SOURCE_DIR = "/home/poyuan/workspace/MCQG/MCQG/data"
    fewshot_path = os.path.join(SOURCE_DIR, "fewshot_source", 'feedback_fewshot.json')
    assert os.path.isfile(fewshot_path)
    rubrics_path = os.path.join(SOURCE_DIR, "input_source", "rubrics.json")
    assert os.path.isfile(rubrics_path)
    input_file_path = os.path.join(SOURCE_DIR, 'inputs', 'inputs_feedback.json')
    feedback_prompt_generator = FeedbackPrompt(fewshot_path, rubrics_path)
    data = json_load(input_file_path)
    qset = data[0]
    feedback_in = get_feedback_in(qset)
    print(feedback_in)
    quit()
    prompt = feedback_prompt_generator.get_prompt(qset, 'context')
    print(prompt.text)
    print("-" * 20)
    reasoning_prompt = feedback_prompt_generator.get_reasoning_prompt(qset, 'reasoning')
    print(reasoning_prompt.text)