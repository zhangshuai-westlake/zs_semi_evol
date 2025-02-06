import sys
sys.path.append(".")
import pandas as pd
import json
from common import *

def format_question_alpaca(row, format_fn=format_multichoice_question, is_pseudo=False):
    input_text = format_fn(row)
    if is_pseudo:
        output_test = f'Answer: {row["PseudoLabel"]}'
    else:
        output_test = f'Answer: {row["answer"]}'

    ######下面代码用于测试
    # output_test = f'Answer: {row["answer"]}'
    # wrong_map = {
    #     "A": "B", "B": "C", "C": "D", "D": "E", "E": "F", "F": "G", "G": "H",
    #     "H": "I", "I": "J", "J": "K", "K": "L", "L": "M", "M": "N", "N": "O",
    # }
    # output_test = f'Answer: {wrong_map[row["answer"]]}'

    output = {
        "instruction": input_text,
        "input": '',
        "output": output_test,
    }
    if is_pseudo:
        output["Accuracy"] = row["Accuracy"]

    return output

def format_qa_gpt(row, format_fn=format_multichoice_question, is_pseudo=False):
    if is_pseudo:
        answer = row["PseudoLabel"]
    else:
        answer = row["answer"]
    return {
        'messages': [
            {"role": "user", "content": format_fn(row)},
            {"role": "assistant", "content": "Answer: " + answer}
        ]
    }

def format_gpt_eval(row, format_fn=format_multichoice_question, is_pseudo=False):
    if is_pseudo:
        answer = row["PseudoLabel"]
    else:
        answer = row["answer"]
    return {
        'question': format_fn(row),
        'answer': f'Answer: {answer}'
    }

if __name__ == '__main__':

    # task = sys.argv[1]
    # datatype = sys.argv[2]
    # task = 'mmlu'
    # type = 'alpaca'
    # datatype = 'labeled'
    # input_file = f'./data/{task}/{datatype}.csv'
    # output_file = f'./sft/data/{task}_{datatype}_{type}.jsonl' if 'gpt' in type else f'./sft/data/{task}_{datatype}_{type}.json'

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    datatype = sys.argv[3]
    type = 'alpaca'

    df = pd.read_csv(input_file)

    is_pseudo = datatype == 'pseudo_labeled'
    if is_pseudo:
        acc = df["Accuracy"].sum() / len(df)
        print(f"before filter acc: {acc}")
        median_value = df["entropy"].median()
        df = df[df['entropy'] <= median_value]
        acc = df["Accuracy"].sum() / len(df)
        print(f"after filter acc: {acc}")

    if type == 'alpaca':
        examples = [format_question_alpaca(row, format_multichoice_question, is_pseudo=is_pseudo) for _, row in df.iterrows()]
        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)
    elif type == 'gpt':
        examples = [format_qa_gpt(row, format_multichoice_question, is_pseudo=is_pseudo) for _, row in df.iterrows()]
        with open(output_file, 'w') as f:
            for obj in examples:
                f.write(json.dumps(obj) + '\n')
    elif type == 'gpt_eval':
        examples = [format_gpt_eval(row, format_multichoice_question, is_pseudo=is_pseudo) for _, row in df.iterrows()]
        with open(output_file, 'w') as f:
            for obj in examples:
                f.write(json.dumps(obj) + '\n')
    
    print(f'Finished formatting {len(examples)} examples to {output_file}')