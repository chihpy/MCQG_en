"""
"""
import os
import json
import re
import random
random.seed(42)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_community.vectorstores import FAISS

#####
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
#####

CONTEXT_SUFFIX = (
    "Generate a context(not the question, in the format Context: ) "
    "based on the given topic from the clinical note :\n"
    "Clinical Note: {clinical_note}\nTopic: {topic}\nKeypoint: {keypoint}"
)

QUESTION_SUFFIX = (
    "Generate a one line question(in the format Question: ) "
    "based on the given context:\nContext: {context}\n"
    "Topic: {topic}\nKeypoint: {keypoint}"
)

ANSWER_SUFFIX = (
    "Generate the correct answer(in the format Correct answer: ) "
    "to the question based on the given context, topic and keypoint"
    "(to which it should be highly related to) :\n"
    "Context: {context}\n Question: {question}\nTopic: {topic}\nKeypoint: {keypoint}"
)

DISTRACTOR_SUFFIX = (
    "Generate distractor options(in the format Distractor options: ) "
    "for the context, question, and correct answer:\n"
    "Context: {context}\nQuestion: {question}\nCorrect answer: {correct_answer}"
)

class qbank_retriever():
    def __init__(self, db_dir, embed_model_name="text-embedding-3-small"):
        self.db_dir = db_dir
        self.embed_model_name = embed_model_name
        self._load_db()

    def _load_db(self):
        self.embed_model = OpenAIEmbeddings(model=self.embed_model_name)
        self.db = FAISS.load_local(self.db_dir, self.embed_model, allow_dangerous_deserialization=True)
    
    def query_retrieve(self, query, k=1, col_name='question'):
        ###
        k=1
        ###
        docs = self.db.similarity_search_with_score(query, k=k) # [(doc, score)]
        rvs = []
        for doc_tuple in docs:
            doc = doc_tuple[0]
            score = doc_tuple[1]
            rv = {}
            rv['score'] = float(score)
            rv[f'{col_name}'] = doc.page_content
            meta = doc.metadata
            rv.update(meta)
            rvs.append(rv)
        return rvs

    def prompt_query_retrieve(self, prompt, query, k=1, col_name='question'):
        query_text = prompt.invoke(query).text
        return self.query_retrieve(query_text, k, col_name)

class QgenInitPrompt():
    def __init__(self, db_dir):
        self.retriever = qbank_retriever(db_dir)
    
    def get_context_prompt(self, query):
        """
        Args:
            query: (dict) with keys: ['clinical_note', 'topic', 'keypoint']
        """
        # retrieve qbank from clinical note
        qbank_from_clinical_note = self.retriever.query_retrieve(query['clinical_note'], k=3)
        context_fewshot_prompt = self.get_fewshot_prompt(qbank_from_clinical_note, CONTEXT_SUFFIX)
        return context_fewshot_prompt.invoke(query)

    def get_question_prompt(self, query, context):
        topic = query['topic']
        keypoint = query['keypoint']
        # retrieve qbank from generated context
        qbank_from_context = self.retriever.query_retrieve(context, k=3)
        question_fewshot_prompt = self.get_fewshot_prompt(qbank_from_context, QUESTION_SUFFIX)
        query = {
            'topic': topic,
            'keypoint': keypoint,
            'context': context
        }
        return question_fewshot_prompt.invoke(query)

    def get_answer_prompt(self, query, context, question):
        context_question = context + '\n' + question
        topic = query['topic']
        keypoint = query['keypoint']
        # retrieve qbank from generated context_question
        qbank_from_context_question = self.retriever.query_retrieve(context_question, k=3)
        question_fewshot_prompt = self.get_answer_fewshot_prompt(qbank_from_context_question, ANSWER_SUFFIX)
        query = {
            'topic': topic,
            'keypoint': keypoint,
            'context': context,
            'question': question
        }
        return question_fewshot_prompt.invoke(query)
    
    def get_distractor_prompt(self, query, context, question, answer):
        context_question = context + '\n' + question
        qa_templ = "Question: {context_question}\nCorrect answer: {correct_answer}"
        qbank_from_answer = self.retriever.query_retrieve(qa_templ.format(context_question=context_question, correct_answer=answer),  3)
        distractor_few_shot_prompt = self.get_distractor_fewshot_prompt(qbank_from_answer, DISTRACTOR_SUFFIX)
        query = {
            'context': context,
            'question': question,
            'correct_answer': answer
        }
        return distractor_few_shot_prompt.invoke(query)

    def get_fewshot_prompt(self, examples, suffix):
        """for context and question
        Args:
            examples
            suffix: prompt defined above
        """
        example_prompt = PromptTemplate(
            input_variables=["question"], 
            template="Context and Question: {question}")
#        example_prompt = PromptTemplate(
#            input_variables=["context", "quest"], 
#            template="Context: {context} \n\nQuestion: {quest}")

        fewshot_prompt = FewShotPromptTemplate(
            examples=examples, 
            example_prompt=example_prompt, 
            suffix=suffix,
            )
        return fewshot_prompt

    def get_answer_fewshot_prompt(self, examples, suffix):
        example_prompt = PromptTemplate(
            input_variables=["question", "correct_answer"], 
            template="Context and Question: {question}\nCorrect answer: {correct_answer}")

        fewshot_prompt = FewShotPromptTemplate(
            examples=examples, 
            example_prompt=example_prompt, 
            suffix=suffix,
        )
        return fewshot_prompt
    
    def get_distractor_fewshot_prompt(self, examples, suffix):
        example_prompt = PromptTemplate(
            input_variables=["question", "correct_answer", "distractor_options"],
            template="Context and Question: {question}\nCorrect answer: {correct_answer}\nDistractor options: {distractor_options}")

        fewshot_prompt = FewShotPromptTemplate(
            examples=examples, 
            example_prompt=example_prompt, 
            suffix=suffix,
        )
        return fewshot_prompt

class QgenInitLgc():
    def __init__(self, fewshot_path, db_dir):
        self._llm_init()
        self.prompt_generator = QgenInitPrompt(db_dir)

    def _llm_init(self):
        model_name='gpt-4.1-mini'
        temperature=0.0
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    def qgen(self, query):
        context = self._gen_context(query)
        question = self._gen_question(query, context)
        #context_question = context + '\n' + question
        answer = self._gen_answer(query, context, question)
        distractor = self._gen_distractor(query, context, question, answer)
        rv = {
            "context": context,
            "question": question,
            "correct_answer": answer,
            'distractor_options': distractor['distractor_lst'],
            'distractor': distractor
        }
        return rv
    
    def qgen_debug(self, query):
        # context
        context_prompt = self.prompt_generator.get_context_prompt(query)
        context_response = self.llm.invoke(context_prompt)
        context = context_response.content
        context_parsed = context.split('Context:')[1].strip()
        # question
        question_prompt = self.prompt_generator.get_question_prompt(query, context_parsed)
        question_response = self.llm.invoke(question_prompt)
        question = question_response.content
        question_parsed = question.split('Question:')[1].strip()
        # answer
        answer_prompt = self.prompt_generator.get_answer_prompt(query, context_parsed, question_parsed)
        answer_response = self.llm.invoke(answer_prompt)
        answer = answer_response.content
        answer_parsed = answer.split('Correct answer:')[1].strip()
        # distractor
        distractor_prompt = self.prompt_generator.get_distractor_prompt(query, context_parsed, question_parsed, answer_parsed)
        distractor_response = self.llm.invoke(distractor_prompt)
        distractor = distractor_response.content
        distractor_lst = self.distractor_parser(distractor)
        options, answer_index = self.options_gen(distractor_lst, answer_parsed)
        rv = {
            'context_prompt': context_prompt.text.split('\n'),
            'question_prompt': question_prompt.text.split('\n'),
            'answer_prompt': answer_prompt.text.split('\n'),
            'distractor_prompt': distractor_prompt.text.split('\n'),
        }
        return rv
    
    def _gen_context(self, query):
        prompt = self.prompt_generator.get_context_prompt(query)
        context_response = self.llm.invoke(prompt)
        context = context_response.content
        # context parser
        context_parsed = context.split('Context:')[1].strip()
        return context_parsed
    
    def _gen_question(self, query, context):
        prompt = self.prompt_generator.get_question_prompt(query, context)
        question_response = self.llm.invoke(prompt)
        question = question_response.content
        question_parsed = question.split('Question:')[1].strip()
        return question_parsed

    def _gen_answer(self, query, context, question):
        prompt = self.prompt_generator.get_answer_prompt(query, context, question)
        answer_response = self.llm.invoke(prompt)
        answer = answer_response.content
        answer_parsed = answer.split('Correct answer:')[1].strip()
        return answer_parsed

    def _gen_distractor(self, query, context, question, answer):
        prompt = self.prompt_generator.get_distractor_prompt(query, context, question, answer)
        distractor_response = self.llm.invoke(prompt)
        distractor = distractor_response.content
        distractor_lst = self.distractor_parser(distractor)
        options, answer_index = self.options_gen(distractor_lst, answer)
        rv = {
            'answer_index': answer_index,
            'options': options,  # include answer
            'distractor_row': distractor,
            'distractor_lst': distractor_lst
        }
        return rv
    
    def distractor_parser(self, distractor):
        """
        convert distractor to list
        """
        lines = distractor.strip().split('\n')
        # 移除第一行 "Distractor options:"
        if lines[0].strip().startswith('Distractor options'):
            lines = lines[1:]
        # 移除開頭的 "A :", "B :" 等，並 strip 掉兩側空白
        options = [line.split(':', 1)[1].strip() for line in lines if ':' in line]
        return options

    def number_to_letter(self, n):
        return chr(ord('A') + n)

    def options_gen(self, distractor_lst, answer):
        options = distractor_lst + [answer]
        random.shuffle(options)
        answer_index = self.number_to_letter(options.index(answer))
        rvs = []
        for idx, option in enumerate(options):
            rvs.append(f"{self.number_to_letter(idx)}: {option}")
        return rvs, answer_index


def json_dump(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(f"Data successfully written to {file_path}")


if __name__ == "__main__":
    # setup input
    clinical_note = (
        "A 58-year-old female with no significant past medical "
        "history presented with shortness of breath, fever, and cough for three days. "
        "The patient received two doses of the COVID vaccine, with the second "
        "dose in May 2021. "
        "In the ER, her vital signs were BP 105/96, HR 131 bpm, RR 20 breaths/min, "
        "oxygen saturation of 96% on room air, "
        "and febrile with a temperature of 102.0°F. "
        "Laboratory assessment is in Table . "
        "Nasopharyngeal swab for SARS-CoV-2 was positive. "
        "CT chest on admission shows no acute infiltrate and nonspecific nodules (Figure ). "
        "The patient was admitted to the general medical ward "
        "and started on antibiotics, dexamethasone, and remdesivir. "
        "The patient developed worsening hypoxia on Day 2, "
        "and CT chest showed widespread airspace disease "
        "throughout the lungs (Figure ). "
        "The patient required 4-5 L per minute via nasal cannula."
    )
    topic = "select most appropriate laboratory or diagnostic study"
    keypoint = "PCR,pulmonary function test"
    query = {
        'clinical_note': clinical_note,
        'topic': topic,
        'keypoint': keypoint,
    }
    # initialization
    db_dir = "data/qbank_embedding/faiss/"
    qgen_init = QgenInitLgc(fewshot_path='', db_dir=db_dir)

#    context = qgen_init._gen_context(query)
#    print("context: ")
#    print(context)
#
#    question = qgen_init._gen_question(query, context)
#    print("question: ")
#    print(question)
#
#    answer = qgen_init._gen_answer(query, context, question)
#    print("answer: ")
#    print(answer)
#
#    distractor = qgen_init._gen_distractor(query, context, question, answer)
#    print("distractor: ")
#    print(distractor)

    save_file_path = os.path.join('/home/poyuan/workspace/MCQG/MCQG', 'data/outputs/test/qgen_init_prompt.json')
    rv = qgen_init.qgen_debug(query)
    json_dump(save_file_path, rv)