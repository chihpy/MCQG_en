"""
"""
import re
import os
import json
import pandas as pd

from langchain_openai import ChatOpenAI
    
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate

class ReasoningAnswerLgc():
    def __init__(self, fewshot_path):
        self.fewshot_prompt = self._get_answer_fewshow_prompt(fewshot_path)
        self._llm_init()
    
    def _llm_init(self):
        model_name='gpt-4.1-mini'
        temperature=0.0
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    def _get_answer_fewshow_prompt(self, fewshot_path):
        template = 'Context: {context}\n\nQuestion: {question}\n\nOptions: {options}\n\nCorrect answer: {correct_answer}\n\nReasoning: {reasoning}'
        example_template = PromptTemplate(
            input_variables=["context", "question", "options", "correct_answer", "reasoning"], 
            template=template
        )
        answer_prompt = (
    #        "Answer the USMLE question and provide a step by step reasoning for reaching that particular answer and rejecting other options. \n\n"
            "Answer the USMLE question in the exact same format as the above examples, "
            "including the 'Correct answer', and 'Reasoning' sections. "
            "Do not add any additional headings or text not shown in the examples.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Options: {options}\n\n"
    #        "Correct answer: "
        )
        #examples = pd.read_json(fewshot_path, lines=True)
        #examples = examples.to_dict(orient='records')
        examples = self._json_load(fewshot_path)

        fewshot_prompt = FewShotPromptTemplate(
            example_prompt=example_template,
            examples=examples,
            suffix=answer_prompt,
        )
        return fewshot_prompt

    def _get_prompt(self, query):
        return self.fewshot_prompt.invoke({
            "question": query['question'],
            'context': query['context'],
            'options': query['options']
        })

    def _json_load(self, file_path):
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        return data_dict
    
    def _output_parser(self, response):
        response = response.content
        try:
            attempted_answer = re.search(
                r"Correct answer:(.*?)(?=\n\n)", 
                response, 
                re.DOTALL
            ).group(1).strip()
            reasoning = re.search(r"Reasoning:(.*)", response, re.DOTALL).group(1).strip()
        except Exception as e:
            print("parser err ", e)
            return {
                "reasoning_answer_response": response
            }
        return {
            "attempted_answer": attempted_answer,
            "reasoning": reasoning,
            "reasoning_answer_response": response
        }
    
    def reasoning_answer(self, query):
        prompt = self._get_prompt(query)
        response = self.llm.invoke(prompt)
        rv = self._output_parser(response)
        return rv