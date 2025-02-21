import os
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from datasets import concatenate_datasets
import json



num_examples = 100
label_ratio, unlabel_ratio, test_ratio = 0.2, 0.6, 0.2
num_label_examples = int(label_ratio * num_examples)
num_unlabel_examples = int(unlabel_ratio * num_examples)
num_test_examples = int(test_ratio * num_examples)

seed = 42

save_root = "data/mix"
os.makedirs(save_root, exist_ok=True)

def save_examples(label_examples, unlabel_examples, test_examples, save_path):
    label_examples.to_csv(f"{save_path}/label.csv", index=False)
    unlabel_examples.to_csv(f"{save_path}/unlabel.csv", index=False)
    test_examples.to_csv(f"{save_path}/test.csv", index=False)

### gsm8k
def transform(example):
    split = "#### "
    splits = example["answer"].split(split)
    assert len(splits) == 2
    return {
        "label": splits[-1],
    }
name = "openai/gsm8k/main"
ds = load_from_disk(f"~/work/data/{name}").shuffle(seed=seed).map(transform)

label_examples = ds["train"].select(range(num_label_examples))
unlabel_examples = ds["train"].select(
    range(num_label_examples, num_label_examples+num_unlabel_examples)
)
test_examples = ds["test"].select(range(num_test_examples))
save_path = os.path.join(save_root, "gsm8k")
save_examples(label_examples, unlabel_examples, test_examples, save_path)
print(f"~~~~~~~~Name:{name}~~~~~~~~~~")
print("--------Q---------")
print(label_examples[0]["question"])
print("--------A---------")
print(label_examples[0]["answer"])



### prm800k
def transform(example):
    old_split = "\n\n# Answer\n\n"
    new_split = "\n#### "
    splits = example["solution"].split(old_split)
    assert len(splits) >= 2
    assert len(set(splits[1:])) == 1
    flag = splits[1] == example["answer"]
    return {
        "question": example["problem"],
        "answer": new_split.join(splits[:2]),
        "label": example["answer"],
        "flag": flag,
    }
train_ds = Dataset.from_json("data/prm800k_train.json")
train_ds = train_ds.shuffle(seed=seed).map(transform).filter(lambda example: example["flag"]).remove_columns(["flag"])
test_ds = Dataset.from_json("data/prm800k_test.json")
test_ds = test_ds.shuffle(seed=seed).rename_column("problem", "question").map(lambda example: {"label": example["answer"]})

label_examples = train_ds.select(range(num_label_examples))
unlabel_examples = train_ds.select(
    range(num_label_examples, num_label_examples+num_unlabel_examples)
)
test_examples = test_ds.select(range(num_test_examples))
save_path = os.path.join(save_root, "prm800k")
save_examples(label_examples, unlabel_examples, test_examples, save_path)
print(f"~~~~~~~~Name:prm800k~~~~~~~~~~")
print("--------Q---------")
print(label_examples[0]["question"])
print("--------A---------")
print(label_examples[0]["answer"])



### mbpp
def transform(example):
    test_list = example["test_list"]
    assert isinstance(test_list, list)
    return {"test_list": json.dumps(test_list)}
name = "google-research-datasets/mbpp"
ds = load_from_disk(f"~/work/data/{name}").shuffle(seed=seed).map(transform)
train_ds = ds["train"].rename_columns({"text": "question", "code": "answer"})
test_ds = ds["test"].rename_columns({"text": "question", "code": "answer"})

label_examples = train_ds.select(range(num_label_examples))
unlabel_examples = train_ds.select(
    range(num_label_examples, num_label_examples+num_unlabel_examples)
)
test_examples = test_ds.select(range(num_test_examples))
save_path = os.path.join(save_root, "mbpp")
save_examples(label_examples, unlabel_examples, test_examples, save_path)
print(f"~~~~~~~~Name:{name}~~~~~~~~~~")
print("--------Q---------")
print(label_examples[0]["question"])
print("--------A---------")
print(label_examples[0]["answer"])


### openai_humaneval
def transform(example):
    question = example["prompt"]
    answer = f"{example['prompt']}{example['canonical_solution']}"
    return {"question": question, "answer": answer}
name = "openai/openai_humaneval"
ds = load_from_disk(f"~/work/data/{name}").shuffle(seed=seed)
test_ds = ds["test"].map(transform)

label_examples = test_ds.select(range(num_label_examples))
unlabel_examples = test_ds.select(
    range(num_label_examples, num_label_examples+num_unlabel_examples)
)
test_examples = test_ds.select(range(num_label_examples+num_unlabel_examples, num_examples))
save_path = os.path.join(save_root, "openai_humaneval")
save_examples(label_examples, unlabel_examples, test_examples, save_path)
print(f"~~~~~~~~Name:{name}~~~~~~~~~~")
print("--------Q---------")
print(label_examples[0]["question"])
print("--------A---------")
print(label_examples[0]["answer"])


### PubMedQA
def transform(example):
    context_obj = example["context"]
    question = "\n".join(
        [f"({name}) {context}" for name, context in zip(context_obj["labels"], context_obj["contexts"])]
    )
    question = question + "\n" + example["question"]
    options, chars = ["yes", "no", "maybe"], ["A", "B", "C"]
    option_description = "\n".join([f"{char}:{option}" for char, option in zip(chars, options)])
    # question = f"Question:\n{question}\nOptions:\n{option_description}"
    label = chars[options.index(example["final_decision"])]
#     answer = example["long_answer"] + "#### " + label
    return {"question": question, "answer": f"Answer: {label}", "label": label, "option_description": option_description}
name = "qiaojin/PubMedQA/pqa_labeled"
train_ds = load_from_disk(f"~/work/data/{name}")["train"]
ds = train_ds.shuffle(seed=seed).map(transform)

label_examples = ds.select(range(num_label_examples))
unlabel_examples = ds.select(
    range(num_label_examples, num_label_examples+num_unlabel_examples)
)
test_examples = ds.select(range(num_label_examples+num_unlabel_examples, num_examples))
save_path = os.path.join(save_root, "PubMedQA")
save_examples(label_examples, unlabel_examples, test_examples, save_path)
print(f"~~~~~~~~Name:{name}~~~~~~~~~~")
print("--------Q---------")
print(label_examples[0]["question"])
print("--------A---------")
print(label_examples[0]["answer"])


### convfinqa
def transform(example):
    question = f"{example['input']}"
    answer = f"Answer: {example['output']}"
    return {"question": question, "answer": answer}

name = "FinGPT/fingpt-convfinqa"
ds = load_from_disk(f"~/work/data/{name}")
ds = ds.map(transform).shuffle(seed=seed)

label_examples = ds["train"].select(range(num_label_examples))
unlabel_examples = ds["train"].select(
    range(num_label_examples, num_label_examples+num_unlabel_examples)
)
test_examples = ds["test"].select(range(num_test_examples))
save_path = os.path.join(save_root, "convfinqa")
save_examples(label_examples, unlabel_examples, test_examples, save_path)
print(f"~~~~~~~~Name:{name}~~~~~~~~~~")
print("--------Q---------")
print(label_examples[0]["question"])
print("--------A---------")
print(label_examples[0]["answer"])


### mmlu
def transform(example):
    question = example["question"]
    options, chars = example["choices"], [chr(65 + i) for i in range(len(example["choices"]))]
    option_description = "\n".join([f"{char}:{option}" for char, option in zip(chars, options)])
    # question = f"Question:\n{question}\nOptions:\n{option_description}"
    label = chars[example["answer_idx"]]
    return {"question": question, "answer": f"Answer: {label}", "label": label, "option_description": option_description}
tasks = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_medicine',
    'college_physics', 'computer_security', 'conceptual_physics', 'econometrics',
    'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts',
    'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
    'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence',
    'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous',
    'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations',
    'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]
name = "cais/mmlu"
dss = []
for task in tqdm(tasks):
    ds = load_from_disk(f"~/work/data/{name}/{task}")
    dss.append(ds)
test_ds = concatenate_datasets(
    [ds["test"] for ds in dss]
).rename_column("answer", "answer_idx").shuffle(seed=seed).map(transform)

label_examples = test_ds.select(range(num_label_examples))
unlabel_examples = test_ds.select(
    range(num_label_examples, num_label_examples+num_unlabel_examples)
)
test_examples = test_ds.select(range(num_test_examples))
save_path = os.path.join(save_root, "mmlu")
save_examples(label_examples, unlabel_examples, test_examples, save_path)
print(f"~~~~~~~~Name:{name}~~~~~~~~~~")
print("--------Q---------")
print(label_examples[0]["question"])
print("--------A---------")
print(label_examples[0]["answer"])


### mmlu_pro
def transform(example):
    question = example["question"]
    options, chars = example["options"], [chr(65 + i) for i in range(len(example["options"]))]
    option_description = "\n".join([f"{char}:{option}" for char, option in zip(chars, options)])
    # question = f"Question:\n{question}\nOptions:\n{option_description}"
    answer = example["answer"]
    return {"question": question, "answer": f"Answer: {answer}", "label": answer, "option_description": option_description}

name = "TIGER-Lab/MMLU-Pro"
ds = load_from_disk(f"~/work/data/{name}")
ds = ds["test"].shuffle(seed=seed).map(transform)

label_examples = ds.select(range(num_label_examples))
unlabel_examples = ds.select(
    range(num_label_examples, num_label_examples+num_unlabel_examples)
)
test_examples = ds.select(range(num_test_examples))
save_path = os.path.join(save_root, "mmlu_pro")
save_examples(label_examples, unlabel_examples, test_examples, save_path)
print(f"~~~~~~~~Name:{name}~~~~~~~~~~")
print("--------Q---------")
print(label_examples[0]["question"])
print("--------A---------")
print(label_examples[0]["answer"])
