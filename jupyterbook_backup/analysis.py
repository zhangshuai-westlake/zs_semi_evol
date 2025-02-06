#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append("..")
import pandas as pd
from common import format_multichoice_question, extract_result
from tqdm import tqdm
from collections import Counter
import numpy as np
from functools import reduce
import json


# In[ ]:


# 验证数据重复

from datasets import load_dataset
ck_ds = load_dataset("cais/mmlu", "clinical_knowledge")
ck_df = pd.DataFrame(ck_ds["test"])
cm_ds = load_dataset("cais/mmlu", "college_medicine")
cm_df = pd.DataFrame(cm_ds["test"])

ck_df["instruction"] = ck_df.apply(
    lambda row: row["question"] + "".join(row["choices"]), axis=1
)

cm_df["instruction"] = cm_df.apply(
    lambda row: row["question"] + "".join(row["choices"]), axis=1
)

pd.merge(
    ck_df, cm_df, on="instruction", how="inner", suffixes=('_unlabed', '_test')
)


# In[2]:


def format_question_alpaca(row, format_fn=format_multichoice_question):
    input_text = format_fn(row)
    output_test = f'Answer: {row["answer"]}'
    return {
        "instruction": input_text,
        "input": '',
        "output": output_test
    }

def format_qa_gpt(row, format_fn=format_multichoice_question):
    return {
        'messages': [
            {"role": "user", "content": format_fn(row)},
            {"role": "assistant", "content": "Answer: " + row["answer"]}
        ]
    }

def append_info(df, subtask_name, instruction_to_row):
    df[subtask_name] = df.apply(lambda row: instruction_to_row[row["instruction"]][subtask_name], axis=1)
    df["from"] = df.apply(lambda row: instruction_to_row[row["instruction"]]["from"], axis=1)
    df["answer"] = df.apply(lambda row: instruction_to_row[row["instruction"]]["answer"], axis=1)


# # 原数据集

# In[3]:


# task = "mmlu"
# subtask_name = "Subject"
task = "mmlu_pro"
subtask_name = "category"

path = f"../data/{task}/labeled.csv"
labeled_raw_df = pd.read_csv(path)
labeled_raw_df["from"] = "labeled"
labeled_raw_df["instruction"] = labeled_raw_df.apply(
    lambda row: format_question_alpaca(row, format_multichoice_question)["instruction"], axis=1
)
labeled_summary_df = labeled_raw_df.groupby(subtask_name).agg(count=('answer', 'size'))
labeled_summary_df["proportion"] = labeled_summary_df.apply(
    lambda row: row["count"] / labeled_summary_df["count"].sum(), axis=1
)


path = f"../data/{task}/unlabeled.csv"
unlabeled_raw_df = pd.read_csv(path)
unlabeled_raw_df["from"] = "unlabeled"
unlabeled_raw_df["instruction"] = unlabeled_raw_df.apply(
    lambda row: format_question_alpaca(row, format_multichoice_question)["instruction"], axis=1
)
unlabeled_summary_df = unlabeled_raw_df.groupby(subtask_name).agg(count=('answer', 'size'))
unlabeled_summary_df["proportion"] = unlabeled_summary_df.apply(
    lambda row: row["count"] / unlabeled_summary_df["count"].sum(), axis=1
)


path = f"../data/{task}/test.csv"
test_raw_df = pd.read_csv(path)
test_raw_df["from"] = "test"
test_raw_df["instruction"] = test_raw_df.apply(
    lambda row: format_question_alpaca(row, format_multichoice_question)["instruction"], axis=1
)
test_summary_df = test_raw_df.groupby(subtask_name).agg(count=('answer', 'size'))
test_summary_df["proportion"] = test_summary_df.apply(
    lambda row: row["count"] / test_summary_df["count"].sum(), axis=1
)
dfs = [
    labeled_summary_df.add_prefix('labeled_'),
    unlabeled_summary_df.add_prefix('unlabeled_'),
    test_summary_df.add_prefix('test_')
]
summary_df = reduce(
    lambda left, right: pd.merge(left, right, on=subtask_name, how='outer', suffixes=('_left', '_right')), dfs
)

raw_df = pd.concat([labeled_raw_df, unlabeled_raw_df, test_raw_df], axis=0, ignore_index=True)

instruction_to_row = {
    format_question_alpaca(row, format_multichoice_question)["instruction"]: row 
    for _, row in tqdm(raw_df.iterrows())
}
unlabel_instruction_to_row = {
    format_question_alpaca(row, format_multichoice_question)["instruction"]: row 
    for _, row in tqdm(unlabeled_raw_df.iterrows())
}
label_instruction_to_row = {
    format_question_alpaca(row, format_multichoice_question)["instruction"]: row 
    for _, row in tqdm(labeled_raw_df.iterrows())
}

overlap_df = pd.merge(
    unlabeled_raw_df, test_raw_df, on="instruction", how="inner", suffixes=('_unlabed', '_test')
)

# overlap_df = pd.merge(
#     labeled_raw_df, test_raw_df, on="instruction", how="inner", suffixes=('_unlabed', '_test')
# )

summary_df
# overlap_df


# # 在unlabeled数据上的表现

# In[4]:


# label_init_df = pd.read_json(f"../sft/data/{task}_labeled_alpaca.json")
# append_info(label_init_df, subtask_name=subtask_name, instruction_to_row=label_instruction_to_row)

unlabeled_warm_df = pd.read_csv(f"../data/{task}/pseudo_warm_llama3.1_{task}.csv")
unlabeled_warm_df["instruction"] = unlabeled_warm_df.apply(
    lambda row: format_question_alpaca(row, format_multichoice_question)["instruction"], axis=1
)
unlabeled_warm_df["Accuracy2"] = unlabeled_warm_df.apply(
    lambda row: int(row["PseudoLabel"] == row["answer"]), axis=1
)
append_info(unlabeled_warm_df, subtask_name=subtask_name, instruction_to_row=unlabel_instruction_to_row)
assert np.all(unlabeled_warm_df["Accuracy"] == unlabeled_warm_df["Accuracy2"])

unlabeled_filter_df = pd.read_json(f"../sft/data/pseudo_warm_llama3.1_{task}_alpaca.json")
append_info(unlabeled_filter_df, subtask_name=subtask_name, instruction_to_row=unlabel_instruction_to_row)
unlabeled_filter_df["Accuracy2"] = unlabeled_filter_df.apply(
    lambda row: int(extract_result(row["output"]) == row["answer"]), axis=1)
assert np.all(unlabeled_filter_df["Accuracy"] == unlabeled_filter_df["Accuracy2"])


unlabeled_warm_summary_df = unlabeled_warm_df.groupby(subtask_name).agg(
    count=('Accuracy', 'size'), accuracy=('Accuracy', 'mean')
)
unlabeled_warm_summary_df["proportion"] = unlabeled_warm_summary_df.apply(
    lambda row: row["count"] / unlabeled_warm_summary_df["count"].sum(), axis=1
)

unlabeled_filter_summary_df = unlabeled_filter_df.groupby(subtask_name).agg(
    count=('Accuracy', 'size'), accuracy=('Accuracy', 'mean')
)
unlabeled_filter_summary_df["proportion"] = unlabeled_filter_summary_df.apply(
    lambda row: row["count"] / unlabeled_filter_summary_df["count"].sum(), axis=1
)

unlabeled_summary_df = pd.merge(
    unlabeled_warm_summary_df.add_prefix("before_"), 
    unlabeled_filter_summary_df.add_prefix("after_"), 
    on=subtask_name, how='outer'
)

for prefix in ["before", "after"]: 
    unlabeled_summary_df[prefix] = unlabeled_summary_df.apply(
        lambda row: f"{row[f'{prefix}_count']}/{round(row[f'{prefix}_proportion'], 4)}", axis=1
    )
    
for prefix in ["before", "after"]:
    unlabeled_summary_df[f"{prefix}_accuracy"] = unlabeled_summary_df.apply(
        lambda row: round(row[f'{prefix}_accuracy'], 3), axis=1
    )
    
unlabeled_show_summary_df = unlabeled_summary_df[[
    "before", "before_accuracy", 
    "after", "after_accuracy", 
]]

unlabeled_show_summary_df.to_csv(f"result/{task}_unlabeled.csv")

unlabeled_show_summary_df


# # 在test集的表现

# In[5]:


result_path = f"../save/{task}__storage_home_westlakeLab_zhangshuai_models_Meta-Llama-3.1-8B-Instruct.json"
base_preds = []
with open(result_path) as fo:
    for line in fo.readlines():
        pred = extract_result(json.loads(line)["response"])
        base_preds.append(pred)

result_path = f"../save/{task}_._sft_output_merged_warm_llama3.1_{task}.json"
warm_preds = []
with open(result_path) as fo:
    for line in fo.readlines():
        pred = extract_result(json.loads(line)["response"])
        warm_preds.append(pred)
        
result_path = f"../save/{task}_._sft_output_merged_pseudo_llama3.1_{task}_filter.json"
semievol_preds = []
with open(result_path) as fo:
    for line in fo.readlines():
        pred = extract_result(json.loads(line)["response"])
        semievol_preds.append(pred)
        
path = f"../data/{task}/test.csv"
test_raw_df = pd.read_csv(path)
test_raw_df["base_pred"] = base_preds
test_raw_df["warm_pred"] = warm_preds
test_raw_df["semievol_pred"] = semievol_preds
test_raw_df["base_accuracy"] = test_raw_df.apply(lambda row: int(row["base_pred"] == row["answer"]), axis=1)
test_raw_df["warm_accuracy"] = test_raw_df.apply(lambda row: int(row["warm_pred"] == row["answer"]), axis=1)
test_raw_df["semievol_accuracy"] = test_raw_df.apply(
    lambda row: int(row["semievol_pred"] == row["answer"]), axis=1
)

test_summary_df = test_raw_df.groupby(subtask_name).agg(
    test_count=(('answer', 'size')),
    base_accuracy=('base_accuracy', 'mean'),
    warm_accuracy=('warm_accuracy', 'mean'),
    semievol_accuracy=('semievol_accuracy', 'mean'),
)
test_summary_df["test_proportion"] = test_summary_df.apply(
    lambda row: row["test_count"] / test_summary_df["test_count"].sum(), axis=1
)

test_summary_df = pd.merge(
    test_summary_df, labeled_summary_df.add_prefix("warm_"), on=subtask_name, how='outer'
)
test_summary_df = pd.merge(
    test_summary_df, 
    unlabeled_warm_summary_df.add_prefix("semievol_before_filter_"), on=subtask_name, how='outer'
)
test_summary_df = pd.merge(
    test_summary_df, 
    unlabeled_filter_summary_df.add_prefix("semievol_after_filter_"), on=subtask_name, how='outer'
)

for prefix in ["test", "warm", "semievol_before_filter", "semievol_after_filter"]: 
    test_summary_df[prefix] = test_summary_df.apply(
        lambda row: f"{row[f'{prefix}_count']}/{round(row[f'{prefix}_proportion'], 4)}", axis=1
    )
    
for prefix in ["base", "warm", "semievol"]:
    test_summary_df[f"{prefix}_accuracy"] = test_summary_df.apply(
        lambda row: round(row[f'{prefix}_accuracy'], 3), axis=1
    )
test_summary_df["after/before"] = round(
    test_summary_df["semievol_after_filter_count"] / test_summary_df["semievol_before_filter_count"], 3
)

test_show_summary_df = test_summary_df[[
    "test", "base_accuracy", 
    "warm", "warm_accuracy", 
    "semievol_before_filter", "semievol_after_filter", "after/before", "semievol_accuracy"
]]

test_show_summary_df.to_csv(f"result/{task}_test.csv")
test_show_summary_df


# In[6]:


test_corr_df = test_show_summary_df[["after/before"]].copy()

assert "".join(test_corr_df.index) == "".join(unlabeled_filter_summary_df.index)
test_corr_df["filter_accuracy"] = round(unlabeled_filter_summary_df["accuracy"], 3)

test_corr_df["delta_accuracy"] = round(
    test_show_summary_df["semievol_accuracy"] - test_show_summary_df["warm_accuracy"], 3
)

test_corr_df = test_corr_df.sort_values(by="after/before", ascending=False)


# # experiment convert data

# ## 数据生成

# In[13]:


### 过滤比例大小处于首尾的subtask以hard的方式往反方向的比例调整

input_file = f"../data/{task}/pseudo_warm_llama3.1_{task}.csv"
pseudo_data_df = pd.read_csv(input_file)

acc = pseudo_data_df["Accuracy"].sum() / len(pseudo_data_df)
print(f"before filter acc: {acc}")


keep_threshold = 11

high_keep_proportion_subtask = test_corr_df[:keep_threshold].index.tolist()
low_keep_proportion_subtask = test_corr_df[-keep_threshold:].index.tolist()


def filter_by_entropy(group):
    subtask = group.name
    if subtask in low_keep_proportion_subtask:
        threshold = group['entropy'].quantile(0.8)
    elif subtask in high_keep_proportion_subtask:
        threshold = group['entropy'].quantile(0.2)
    else:
        threshold = group['entropy'].quantile(0.5)
    return group[group['entropy'] < threshold]

filter_pseudo_data_df = pseudo_data_df.groupby(subtask_name, group_keys=False).apply(filter_by_entropy)

acc = filter_pseudo_data_df["Accuracy"].sum() / len(filter_pseudo_data_df)
print(f"after filter acc: {acc}")
print(f"after filter number example: {len(filter_pseudo_data_df)}")


# In[10]:


### 按类取中位数

input_file = f"../data/{task}/pseudo_warm_llama3.1_{task}.csv"
pseudo_data_df = pd.read_csv(input_file)

acc = pseudo_data_df["Accuracy"].sum() / len(pseudo_data_df)
print(f"before filter acc: {acc}")

filter_pseudo_data_df = pseudo_data_df.groupby(subtask_name, group_keys=False).apply(
    lambda group: group[group["entropy"] < group['entropy'].median()]
)

acc = filter_pseudo_data_df["Accuracy"].sum() / len(filter_pseudo_data_df)
print(f"after filter acc: {acc}")
print(f"after filter number example: {len(filter_pseudo_data_df)}")


# In[64]:


### 两头挤

input_file = f"../data/{task}/pseudo_warm_llama3.1_{task}.csv"
pseudo_data_df = pd.read_csv(input_file)

acc = pseudo_data_df["Accuracy"].sum() / len(pseudo_data_df)
print(f"before filter acc: {acc}")

weight = 0.5
overall_median = pseudo_data_df["entropy"].median()
filter_pseudo_data_df = pseudo_data_df.groupby(subtask_name, group_keys=False).apply(
    lambda group: group[group["entropy"] < weight * group['entropy'].median() + (1 - weight) * overall_median]
)

acc = filter_pseudo_data_df["Accuracy"].sum() / len(filter_pseudo_data_df)
print(f"after filter acc: {acc}")
print(f"after filter number example: {len(filter_pseudo_data_df)}")


# In[18]:


### 直接取groud truth看模型上限

input_file = f"../data/{task}/pseudo_warm_llama3.1_{task}.csv"
pseudo_data_df = pd.read_csv(input_file)

acc = pseudo_data_df["Accuracy"].sum() / len(pseudo_data_df)
print(f"before filter acc: {acc}")

filter_pseudo_data_df = pseudo_data_df.copy()
filter_pseudo_data_df["PseudoLabel"] = pseudo_data_df["answer"]
filter_pseudo_data_df["Accuracy"] = 1.0

acc = filter_pseudo_data_df["Accuracy"].sum() / len(filter_pseudo_data_df)
print(f"after filter acc: {acc}")
print(f"after filter number example: {len(filter_pseudo_data_df)}")


# In[12]:


def format_question_alpaca(row, format_fn=format_multichoice_question):
    input_text = format_fn(row)
    output_test = f'Answer: {row["PseudoLabel"]}'

    return {
        "instruction": input_text,
        "input": '',
        "output": output_test,
        "Accuracy": row["Accuracy"],
    }

examples = [
    format_question_alpaca(row, format_multichoice_question) 
    for _, row in filter_pseudo_data_df.iterrows()
]

output_file=f"../sft/data/pseudo_warm_llama3.1_{task}_alpaca_threshold_by_filter_proportion.json"

with open(output_file, 'w') as f:
    json.dump(examples, f, indent=2)


# ## 新数据下模型的表现

# In[7]:


result_path = f"../save/{task}_._sft_output_merged_pseudo_llama3.1_{task}_filter_threshold_by_filter_proportion.json"

semievol_threshold_preds = []
with open(result_path) as fo:
    for line in fo.readlines():
        pred = extract_result(json.loads(line)["response"])
        semievol_threshold_preds.append(pred)

test_threshold_df = test_raw_df.copy()[[subtask_name, "answer"]]
test_threshold_df["semievol_threshold_pred"] = semievol_threshold_preds
test_threshold_df["semievol_threshold_accuracy"] = test_threshold_df.apply(
    lambda row: int(row["answer"] == row["semievol_threshold_pred"]), axis=1
)
test_threshold_summary_df = test_threshold_df.groupby(subtask_name).agg( 
    count=('semievol_threshold_accuracy', 'size'), 
    semievol_threshold_accuracy=('semievol_threshold_accuracy', 'mean')  
)

unlabeled_threshold_filter_df = pd.read_json(
    f"../sft/data/pseudo_warm_llama3.1_{task}_alpaca_threshold_by_filter_proportion.json"
)
append_info(
    unlabeled_threshold_filter_df, subtask_name=subtask_name, instruction_to_row=unlabel_instruction_to_row
)
unlabeled_threshold_filter_summary_df = unlabeled_threshold_filter_df.groupby(subtask_name).agg(
    count=('Accuracy', 'size'), accuracy=('Accuracy', 'mean')
)


# In[8]:


merge_df = pd.merge(
    unlabeled_threshold_filter_summary_df, unlabeled_warm_summary_df, 
    on=subtask_name, how="inner"
)
test_threshold_corr_df = pd.DataFrame({"after/before": round(merge_df["count_x"] / merge_df["count_y"], 3)})

assert "".join(test_threshold_corr_df.index) == "".join(unlabeled_threshold_filter_summary_df.index)
test_threshold_corr_df["filter_accuracy"] = round(unlabeled_threshold_filter_summary_df["accuracy"], 3)

merge_df = pd.merge(
    test_show_summary_df, test_threshold_summary_df, 
    on=subtask_name, how="inner"
)
assert "".join(merge_df.index) == "".join(test_threshold_corr_df.index)
test_threshold_corr_df[
    "delta_accuracy"
] = round(merge_df["semievol_threshold_accuracy"] - merge_df["warm_accuracy"], 4)


# ## 新旧模型表现对比

# In[9]:


test_corr_compare_df = pd.merge(
    test_corr_df, test_threshold_corr_df, on=subtask_name, how="left", suffixes=("###baseline", "###our")
)
test_corr_compare_df.to_csv(f"result/{task}_test_compare.csv")
test_corr_compare_df

