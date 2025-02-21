set -ex


### warm sft make data
#datatype=labeled
#input_file="./data/mix"
#output_file="./sft/data/mix_labeled_alpaca.json"



base_model="llama3.1"
#base_model="tinyllama1.1base"
#base_model="phi3mini4k"
#base_model="gemma2_2bit"
#base_model="gemma2_2b"

### pseudo make data
datatype=pseudo_labeled
input_file="./data/mix/pseudo_warm_${base_model}_mix.csv"
output_file="./sft/data/pseudo_warm_${base_model}_mix_alpaca.json"


python utils/mix_convert_data.py $input_file $output_file $datatype