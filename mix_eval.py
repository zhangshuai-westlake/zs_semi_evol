import random
import re
import pandas as pd
import sys
sys.path.append('..')
import json
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
from common import *
from utils import *
from config import *
import datetime
import re
import json
from copy import deepcopy
import openai_humaneval_execution
import mbpp_execution

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mix_template import *


def str_to_bool(s):
    s = s.strip().lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert {s} to boolean")


class MathEvaluator:
    ## gsm8k, prm800k
    def __init__(self):
        model_path = "/storage/home/westlakeLab/zhangshuai/models/KbsdJames/Omni-Judge"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # set terminators for decoding
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def __call__(self, question, pred_answer, reference):
        formatted_context = self.tokenizer.get_context(
            question,
            str(reference),
            pred_answer,
        )
        model_inputs = self.tokenizer(formatted_context, return_tensors="pt")
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # do inference
        pred = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
            do_sample=False,
            num_return_sequences=1,
            max_new_tokens=300,
        )[0].cpu().tolist()

        # post-process
        pred = pred[len(input_ids[0].cpu().tolist()):]
        for terminator in self.terminators:
            if terminator in pred:
                pred = pred[:pred.index(terminator)]
        response = self.tokenizer.decode(pred, skip_special_tokens=True)
        pred_truth = self.tokenizer.parse_response(response)

        return str_to_bool(pred_truth["judgement"])


def check(example, math_evaluator=None):
    task = example["task"]
    response = example["response"]
    # response = example["answer"]

    if task in ("PubMedQA", "mmlu", "mmlu_pro"):
        answer = example["label"]
        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-Z])"
        match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
        pred_answer = match.group(1) if match else response[0].upper()
        return pred_answer == answer

    if task == "gsm8k":
        question = example["question"]
        pred_answer = response.split("####")[-1].strip()
        reference = example["label"]
        return math_evaluator(question, pred_answer, reference)

    if task == "prm800k":
        question = example["question"]
        pred_answer = response.split("####")[-1].strip()
        reference = example["label"]
        return math_evaluator(question, pred_answer, reference)

    if task == "convfinqa":
        question = example["question"]
        ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
        match = re.search(ANSWER_PATTERN, response)
        pred_answer = match.group(1) if match else response[0].upper()
        reference = example["output"]
        return math_evaluator(question, pred_answer, reference)

    if task == "openai_humaneval":
        response = example['response']
        code_responses = re.findall(f'```python\n(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if len(code_responses) > 0:
            response = code_responses[0]
        result = openai_humaneval_execution.check_correctness(example, completion=response, timeout=10)
        return result["passed"]

    if task == "mbpp":
        excution_example = deepcopy(example)
        response = example['response']
        code_responses = re.findall(f'```python\n(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if len(code_responses) > 0:
            response = code_responses[0]
        test_list = json.loads(example["test_list"])
        excution_example["test_code"] = response + "\n" + "\n".join(test_list)
        result = mbpp_execution.check_correctness(excution_example["task_id"], excution_example, "python")
        return result["passed"]

    raise ValueError(f"Task {task} not support")

def format_zero_shot(example):
    task = example["task"]
    if task in ("PubMedQA", "mmlu", "mmlu_pro"):
        content = ZS_MULTICHOICE_TEMPLATE.format(question=example["question"], options=example["option_description"])
    elif task == "gsm8k":
        content = ZS_GSM8K_TEMPLATE.format(question=example["question"])
    elif task == "openai_humaneval":
        content = ZS_OPENAI_HUMANEVAL_TEMPLATE.format(question=example["question"])
    elif task == "convfinqa":
        content = ZS_CONVFINQA_TEMPLATE.format(question=example["question"], instruction=example["instruction"])
    elif task == "mbpp":
        test_list = json.loads(example["test_list"])
        content = ZS_MBPP_TEMPLATE.format(question=example["question"], tests="\n".join(test_list))
    elif task == "prm800k":
        content = ZS_PRM800K_TEMPLATE.format(question=example["question"])
    else:
        raise ValueError(f"Task {task} not support")

    return [{"role": "user", "content": content}]


class Evaluator:

    def __init__(self, examples, config=None, suffix=""):
        self.config = config
        self.examples = examples

        model = config['model'].replace('/', '_')
        self.output_path = f'./save/mix_{model}'

        if suffix:
            self.output_path += "_" + suffix
        self.output_path += ".json"
        self.gptreq = None

        self.math_evaluator = MathEvaluator()

    def inference(self, instances):
        if not self.gptreq:
            self.gptreq = LoopRequest()
        res_list = self.gptreq.batch_req(instances, self.config)

        assert len(res_list) == len(self.examples)

        for i, s in enumerate(self.examples):
            response = res_list[i]['response']
            self.examples[i]["message"] = instances[i]
            self.examples[i]["response"] = response
            if "logprobs" in res_list[i]:
                self.examples[i]["logprobs"] = res_list[i]["logprobs"]


    def extract_results(self):
        return self.examples

    def eval(self, format_fn=None):
        print(f'Formating questions ...')
        instances = []
        for example in tqdm(self.examples):
            if format_fn is not None:
                instance = format_fn(example)
            else:
                instance = [{"role": "user", "content": example["question"]}]
            instances.append(instance)

        print(f'Begin Inference ...')
        # return 0
        self.inference(instances)

        checks = []
        print(f'Begin checking ...')
        for i, example in enumerate(self.examples):
            flag = check(example, math_evaluator=self.math_evaluator)
            example["check"] = flag
            checks.append(int(flag))

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as file:
            for example in self.examples:
                file.write(json.dumps(example) + '\n')

        acc = np.mean(checks)
        return acc

    def get_examples(self):
        return self.examples


if __name__ == "__main__":
    math_evaluator = MathEvaluator()

    global_res_file = 'global_res.txt'
    model = sys.argv[1]
    port = sys.argv[2]
    num_examples = None
    if len(sys.argv) > 3:
        num_examples = int(sys.argv[3])
    os.environ['LLM_BASE_URL'] = f"http://localhost:{port}/v1"

    assert model in MODELS_CONFIG

    model_config = MODELS_CONFIG[model]
    infer_config = {
        "model": model_config["name"],
        # "temperature": 0.5,
        "temperature": 0.0,
        "max_tokens": 1000,
        "logprobs": True
    }

    tasks = [
        "PubMedQA", "gsm8k", "mmlu", "openai_humaneval", "convfinqa", "mbpp", "mmlu_pro", "prm800k"
    ]
    data_root = "data/mix"
    examples = []
    for task in tasks:
        task_examples = pd.read_csv(os.path.join(data_root, task, "test.csv")).to_dict('records')
        if num_examples is not None:
            task_examples = task_examples[:num_examples]
        for ex in task_examples:
            ex["task"] = task
        examples.extend(task_examples)

    eval = Evaluator(examples, config=infer_config)

    acc = eval.eval(format_fn=format_zero_shot)

    print(f'Accuracy: {acc}')