set -ex

path="data/mmlu/pseudo_warm_llama3.1_mmlu.csv"
saved_path="sft/data/pseudo_warm_llama3.1_mmlu_filter.json"

python filter_data.py $path $saved_path