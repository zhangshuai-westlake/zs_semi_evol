import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import numpy as np
from string import Template


qa_template = """Here is a following question you need to answer. Your answer should be one in the following options.
Question: {question}
Options: {option_description}
Answer:  """
# qa_template = """Question: {question}
# Options: {option_description}
# Answer: 
# """

def generate_chars(n):
    return [chr(65 + i) for i in range(n)]

def cal_ppl(model, tokenizer, q_column_name, a_column_name, example, option_column_name, option_description, instruction_column_name, do_log=False):
    assert option_column_name is None or option_description is None
    with torch.no_grad():
        q = example[q_column_name]
        if option_column_name is not None:
            options = example[option_column_name]
            assert isinstance(options, list)
            chars = generate_chars(len(options))
            option_description = ";".join([f"{char}: {option}" for char, option in zip(chars, options)])
        if option_description is not None:
            q = qa_template.format(question=q, option_description=option_description)
        if instruction_column_name is not None:
            q = example[instruction_column_name] + q
        a = example[a_column_name]
        if isinstance(a, int) and option_column_name is not None:
            a = chars[a]
        if do_log:
            print("####### Here is question #######")
            print(q)
            print("####### Here is answer #######")
            print(a)
        input_id = tokenizer.encode(q + a, return_tensors="pt").cuda()
        label = input_id.clone().cuda()
        label[0][:len(tokenizer.encode(q))] = -100
        # a = label.clone()[0]
        # print(tokenizer.decode(a[a != -100]))
        o = model(input_ids=input_id, labels=label)
        ppl = o.loss.exp()
    return ppl.item()


def main(
        model_path: str,
        ds_path: str,
        ds_split: str,
        num_examples: int,
        q_column_name: str,
        a_column_name: str,
        ds_name: str = None,
        seed: int = 42,
        option_column_name: str = None,
        option_description: str = None, 
        instruction_column_name: str = None,
):
    print(locals())

    ds = load_dataset(ds_path, ds_name)

    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    ds = ds.shuffle(seed=seed)

    ppls = []
    for i in tqdm(range(num_examples)):
        example = ds[ds_split][i]
        ppl = cal_ppl(model, tokenizer, q_column_name, a_column_name, example, option_column_name, option_description, instruction_column_name, do_log=i==0)
        ppls.append(ppl)

    mean_ppl = np.mean(ppls)
    print(f"mean ppl: {mean_ppl}")


if __name__ == '__main__':
    from fire import Fire
    Fire(main)