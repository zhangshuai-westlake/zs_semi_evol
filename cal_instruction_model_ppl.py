import os
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import numpy as np
from openai import OpenAI
import re


QUERY_TEMPLATE_MULTICHOICE = """
Answer the multiple choice question. Your response should be of the following format: 'Answer: LETTER' (without quotes).

Question: 
{question}

Options:
{option_description}

""".strip()

def paser_options(options):
    if "|" in options:
        options = options.split("|")
    else:
        options = json.loads(options)
    assert isinstance(options, list)
    return options

def format_options(example, option_column_name, options):
    assert option_column_name is None or options is None
    if option_column_name is not None:
        options = example[option_column_name]
    if type(options) == str:
        options = paser_options(options)
    assert isinstance(options, list)
    option_description = [f"{chr(65+i)}. {option}" for i, option in enumerate(options)]
    option_description = "\n".join(option_description)
    return option_description

def format_multichoice(example, q_column_name, option_column_name, options):
    system_prompt = "You are an expert in the multiple choice question."
    question = example[q_column_name]
    option_description = format_options(example, option_column_name, options)
    user_prompt = QUERY_TEMPLATE_MULTICHOICE.format(question=question, option_description=option_description)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def get_ppl(objs):
    ppl = np.exp(-np.mean([obj.logprob for obj in objs]))
    return ppl

def extract_result(ans):
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-Z])"
    match = re.search(ANSWER_PATTERN_MULTICHOICE, ans)
    extracted_answer = match.group(1) if match else ans[0].upper()
    return extracted_answer
def cal_ppl(
        model_path, model_port,
        example,
        q_column_name, a_column_name,
        option_column_name, options,
        temperature=1.0,
        do_log=False,
):
    base_url = f"http://localhost:{model_port}/v1"
    client = OpenAI(api_key="LOCAL_KEY", base_url=base_url)
    msg = format_multichoice(example, q_column_name, option_column_name, options)
    gt_ans = example[a_column_name]
    if options is not None:
        option_list = paser_options(options)
        gt_ans = option_list.index(gt_ans)
    if isinstance(gt_ans, int):
        gt_ans = f"{chr(65+gt_ans)}"
    if do_log:
        print("======= Here is the message ======")
        print(msg[1]["content"])
        print("======= Here is the answer ======")
        print(gt_ans)

    chat_completion = client.chat.completions.create(
        messages=msg, max_tokens=100, logprobs=True, model=model_path, temperature=temperature,
    )
    response = chat_completion.choices[0].message.content
    pred_ans = extract_result(response)
    check = pred_ans == gt_ans
    logprob_obj = chat_completion.choices[0].logprobs.content
    ppl = get_ppl(logprob_obj)

    if do_log:
        print("======= Here is the response ======")
        print(response)
        print("======= Here is the extracted answer ======")
        print(pred_ans)
        # print("======= Here is the logprobs ======")
        # print(logprob_obj)

    return ppl, check


def main(
        model_path: str,
        model_port: int,
        ds_path: str,
        ds_split: str,
        num_examples: int,
        q_column_name: str,
        a_column_name: str,
        ds_name: str = None,
        option_column_name: str = None,
        options: str = None,
        temperature: float = 1.0,
        seed: int = 42,
):
    print(locals())

    ds = load_dataset(ds_path, ds_name)
    ds = ds.shuffle(seed=seed)

    ppls, checks = [], []
    for i in tqdm(range(num_examples)):
        example = ds[ds_split][i]
        ppl, check = cal_ppl(
            model_path, model_port,
            example,
            q_column_name, a_column_name,
            option_column_name, options,
            temperature=temperature,
            do_log=i==0
        )
        ppls.append(ppl)
        checks.append(check)
    ppls, checks = np.array(ppls), np.array(checks)

    mean_ppl = ppls.mean()

    acc = checks.sum() / len(checks)

    threshold = np.median(ppls)
    filter_checks = checks[ppls < threshold]
    filter_acc = filter_checks.sum() / len(filter_checks)


    print(f"acc: {acc}")
    print(f"filter_acc: {filter_acc}")
    print(f"mean ppl: {mean_ppl}")


if __name__ == '__main__':
    from fire import Fire
    Fire(main)