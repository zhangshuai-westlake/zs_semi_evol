import random
import pandas as pd
import os
import sys
sys.path.append('..')
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
import copy
import numpy as np
from mix_eval import Evaluator
import threading
from config import *
from mix_retrieval import NearestReference
from mix_template import *

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

def calculate_entropy(probs):
    prob_list = np.array(probs)
    entropy = - np.sum(prob_list) / len(prob_list)
    return entropy

GLOBAL_RETRIEVAL_CACHE = ThreadSafeDict()

def format_few_shot(example, topk=None):
    reference = nr.fewshot(example)
    if topk is not None:
        elems = reference.split("Example")
        assert len(elems) == 4
        reference = "\n".join(elems[:1+topk])
    task = example["task"]
    if task in ("PubMedQA", "mmlu", "mmlu_pro"):
        content = FS_MULTICHOICE_TEMPLATE.format(
            question=example["question"], options=example["option_description"], reference=reference,
        )
    elif task == "gsm8k":
        content = FS_GSM8K_TEMPLATE.format(question=example["question"], reference=reference)
    elif task == "openai_humaneval":
        content = FS_OPENAI_HUMANEVAL_TEMPLATE.format(question=example["question"], reference=reference)
    elif task == "convfinqa":
        content = FS_CONVFINQA_TEMPLATE.format(
            question=example["question"], instruction=example["instruction"], reference=reference
        )
    elif task == "mbpp":
        test_list = json.loads(example["test_list"])
        content = FS_MBPP_TEMPLATE.format(
            question=example["question"], tests="\n".join(test_list), reference=reference
        )
    elif task == "prm800k":
        content = FS_PRM800K_TEMPLATE.format(question=example["question"], reference=reference)
    else:
        raise ValueError(f"Task {task} not support")

    return [{"role": "user", "content": content}]

if __name__ == '__main__':
    model = sys.argv[1]
    port = sys.argv[2]
    if len(sys.argv) < 4:
        num_examples = None
    else:
        num_examples = int(sys.argv[3])

    def load_examples(tasks, split, num_examples=None):
        data_root = "data/mix"
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
    label_examples = load_examples(tasks, "label", num_examples=num_examples)
    unlabel_examples = load_examples(tasks, "unlabel", num_examples=num_examples)

    infer_config = {
        'config': {
            "model": MODELS_CONFIG[model]["name"],
            "temperature": 1,
            "max_tokens": 2000,
            "logprobs": True
        }
    }
    os.environ['LLM_BASE_URL'] = f"http://localhost:{port}/v1"


    nr = NearestReference(k=3)
    nr.build(label_examples)

    evaluator = Evaluator(examples=unlabel_examples, **infer_config, suffix="fewshot")
    few_shot_acc = evaluator.eval(format_fn=format_few_shot)
    print(f"Few-shot accuracy: {few_shot_acc}")

    save_data = copy.deepcopy(evaluator.get_examples())
    for example in save_data:
        pred_list = []
        example['PseudoLabel'] = example['response']
        entropy = calculate_entropy(example['logprobs'])
        example['entropy'] = entropy

    save_df = pd.DataFrame(save_data)
    save_path = f'data/mix/pseudo_{model}.csv'
    save_df.to_csv(save_path, index=False)
    print(f"Save the final pseudo-labels to {save_path}")
