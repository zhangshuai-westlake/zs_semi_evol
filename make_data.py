import os
import pandas as pd

task = "mmlu"
subtask = "Subject"

# task = "mmlu_pro"
# subtask = "category"

sample_fraction = 0.25
seed = 42

origin_data_dir = f"./data/{task}_origin"
data_dir = f"./data/{task}"
labeled_df = pd.read_csv(f"{origin_data_dir}/labeled.csv")

labeled_df_sampled = labeled_df.groupby(subtask, group_keys=False).apply(
    lambda x: x.sample(frac=sample_fraction, random_state=seed)
)

labeled_df_sampled.to_csv(f"{data_dir}/labeled.csv", index=False)