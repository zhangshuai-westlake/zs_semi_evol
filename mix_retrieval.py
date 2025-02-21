import os
import sys
import json
import numpy as np
import pandas as pd
import threading
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from mix_template import *
from tqdm import tqdm

class ThreadSafeDict:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}

    def get(self, key):
        with self.lock:
            return self.data.get(key)

    def set(self, key, value):
        with self.lock:
            self.data[key] = value

GLOBAL_RETRIEVAL_CACHE = ThreadSafeDict()

def format_question_and_answer(example):
    task = example["task"]
    if task in ("PubMedQA", "mmlu", "mmlu_pro"):
        content = QA_MULTICHOICE_TEMPLATE.format(
            question=example["question"], options=example["option_description"],
            answer=example["answer"],
        )
    elif task == "gsm8k":
        content = QA_GSM8K_TEMPLATE.format(question=example["question"], answer=example["answer"])
    elif task == "openai_humaneval":
        content = QA_OPENAI_HUMANEVAL_TEMPLATE.format(question=example["question"], answer=example["answer"])
    elif task == "convfinqa":
        content = QA_CONVFINQA_TEMPLATE.format(question=example["question"], answer=example["answer"])
    elif task == "mbpp":
        test_list = json.loads(example["test_list"])
        content = QA_MBPP_TEMPLATE.format(question=example["question"], tests="\n".join(test_list), answer=example["answer"])
    elif task == "prm800k":
        content = QA_PRM800K_TEMPLATE.format(question=example["question"], answer=example["answer"])
    else:
        raise ValueError(f"Task {task} not support")

    return content


class NearestReference:
    def __init__(self, k=3) -> None:
        self.vectorstore = None
        self.selector = None
        self.k = k
        self.embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.embed_path = f'tmp/mix_labeled'
        self.search_path = f"{self.embed_path}/search_idx_k={k}.json"
        self.search_map = {}
        if os.path.exists(self.search_path):
            self.search_map = {item["query"]: item["ref_str"] for item in json.load(open(self.search_path))}

    def build(self, label_examples):
        embed_path = self.embed_path
        if os.path.exists(embed_path) and os.path.exists(f'{embed_path}/index.faiss'):
            self.vectorstore = FAISS.load_local(embed_path, self.embed_model, allow_dangerous_deserialization=True)
        else:
            texts = [example['question'] for example in label_examples]
            self.vectorstore = FAISS.from_texts(texts, self.embed_model, metadatas=label_examples)
            os.makedirs(embed_path, exist_ok=True)
            self.vectorstore.save_local(embed_path)

        self.selector = SemanticSimilarityExampleSelector(
            vectorstore=self.vectorstore, k=self.k
        )
        return self.vectorstore

    def fewshot(self, example):
        question = example['question']
        if question in self.search_map:
            return self.search_map[question]
        ref = self.retrieve(question)

        ref_str = ''
        for i, example in enumerate(ref):
            ref_str += f"Example {i+1}:\n{format_question_and_answer(example)}\n\n"
        self.search_map[question] = ref_str
        return ref_str

    # @retry
    # @func_set_timeout(2)
    def retrieve(self, question):
        cached_result = GLOBAL_RETRIEVAL_CACHE.get(question)
        if cached_result is not None:
            return cached_result
        res = self.selector.select_examples({'question': question})
        GLOBAL_RETRIEVAL_CACHE.set(question, res)
        return res


    def save_search_map(self):
        if not os.path.exists(self.search_path):
            items = [{'query': k, 'ref_str': v} for k, v in self.search_map.items()]
            with open(self.search_path, 'w') as f:
                json.dump(items, f, indent=4)


if __name__ == '__main__':
    k = 3
    nr = NearestReference(k=k)
    def load_examples(tasks, split, num_examples=None):
        # data_root = "data/mix"
        data_root = "remote_data/mix"
        examples = []
        for task in tasks:
            task_examples = pd.read_csv(os.path.join(data_root, task, f"{split}.csv")).to_dict('records')
            if num_examples is not None:
                task_examples = task_examples[:num_examples]
            for ex in task_examples:
                ex["task"] = task
            examples.extend(task_examples)
        return examples

    tasks = [
        "PubMedQA", "gsm8k", "mmlu", "openai_humaneval", "convfinqa", "mbpp", "mmlu_pro", "prm800k"
    ]

    # num_examples = 5
    num_examples = None
    label_examples = load_examples(tasks, "label", num_examples=num_examples)
    unlabel_examples = load_examples(tasks, "unlabel", num_examples=num_examples)


    print("Building vectorstore...")
    nr.build(label_examples)

    print("Retrieving...")
    for example in tqdm(unlabel_examples):
        print("======= Here is question =======")
        print(example["question"])
        print("======= Here is fewshot =======")
        fewshot_str = nr.fewshot(example)
        print(fewshot_str)

    print("Saving search map...")
    nr.save_search_map()