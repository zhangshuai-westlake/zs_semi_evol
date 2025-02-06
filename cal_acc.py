import os
import sys
import json
import pandas as pd
from common import extract_result

def cal_acc(result_path, df):
    preds = []
    answers = df["answer"]
    with open(result_path) as fo:
        for line in fo.readlines():
            item = json.loads(line)
            pred = extract_result(item["response"])
            preds.append(pred)
    assert len(preds) == len(answers)
    checks = [pred == ans for pred, ans in zip(preds, answers)]
    return sum(checks) / len(checks)

# task = "mmlu"
# task = "USMLE"
# path = f"./data/{task}/test.csv"
# print(path)



# ### base instruction model
# # name = "USMLE__storage_home_westlakeLab_zhangshuai_models_Meta-Llama-3.1-8B-Instruct"
# # name = "mmlu__storage_home_westlakeLab_zhangshuai_models_Meta-Llama-3.1-8B-Instruct"
#
# ### warm sft model
# # name = "USMLE_._sft_output_merged_warm_llama3.1_USMLE"
#
# ### semievol
# # name = "USMLE__storage_home_westlakeLab_zhangshuai_work_semi_instruction_tuning_SemiEvol_sft_output_merged_pseudo_llama3.1_USMLE_filter"
# # name = "mmlu__storage_home_westlakeLab_zhangshuai_work_semi_instruction_tuning_SemiEvol_sft_output_merged_pseudo_llama3.1_mmlu_filter"
# name = "mmlu__storage_home_westlakeLab_zhangshuai_work_semi_instruction_tuning_SemiEvol_sft_output_merged_pseudo_warm_llama3.1_mmlu_filter"
# result_path = f"./save/{name}.json"

path, result_path = sys.argv[1], sys.argv[2]

df = pd.read_csv(path)

print(cal_acc(result_path, df))