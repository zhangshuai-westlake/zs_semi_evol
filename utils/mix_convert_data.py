import os
import sys
sys.path.append(".")
import pandas as pd
import json
from mix_eval import format_zero_shot

def format_question_alpaca(example, is_pseudo=False):
    input_text = format_zero_shot(example)[0]["content"]
    output_test = example["PseudoLabel"] if is_pseudo else example["answer"]

    output = {
        "instruction": input_text,
        "input": '',
        "output": output_test,
        "task": example["task"],
    }
    if is_pseudo:
        output["check"] = example["check"]

    return output

def format_qa_gpt(example, is_pseudo=False):
    input_text = format_zero_shot(example)[0]["content"]
    output_test = example["PseudoLabel"] if is_pseudo else example["answer"]
    return {
        'messages': [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_test}
        ]
    }


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    datatype = sys.argv[3]

    def load_df(tasks, num_examples=None):
        split = "label"
        data_root = input_path
        dfs = []
        for task in tasks:
            task_df = pd.read_csv(os.path.join(data_root, task, f"{split}.csv"))
            if num_examples is not None:
                task_df = task_df[:num_examples]
            task_df["task"] = task
            dfs.append(task_df)
        df = pd.concat(dfs, ignore_index=True)
        return df

    is_pseudo = datatype == 'pseudo_labeled'
    if is_pseudo:
        df = pd.read_csv(input_path)
        acc = df["check"].mean()
        print(f"before filter acc: {acc}")
        median_value = df["entropy"].median()
        df = df[df['entropy'] <= median_value]
        acc = df["check"].mean()
        print(f"after filter acc: {acc}")
    else:
        tasks = [
            "PubMedQA", "gsm8k", "mmlu", "openai_humaneval", "convfinqa", "mbpp", "mmlu_pro", "prm800k"
        ]
        # df = load_df(tasks, num_examples=5)
        df = load_df(tasks)


    examples = [format_question_alpaca(example, is_pseudo=is_pseudo) for _, example in df.iterrows()]

    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f'Finished formatting {len(examples)} examples to {output_path}')