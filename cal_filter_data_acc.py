import os
import sys
import json
import pandas as pd
from common import format_multichoice_question, format_question_and_answer

path = sys.argv[1]

# path = "data/mmlu/pseudo_warm_llama3.1_mmlu.csv"

df = pd.read_csv(path)
print("Before filter accuracy")
print(df["Accuracy"].sum() / len(df))


print("After filter accuracy")
median_value = df["entropy"].median()
filtered_df = df[df['entropy'] <= median_value]
print("Accuracy")
print(filtered_df["Accuracy"].sum() / len(filtered_df))