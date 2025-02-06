set -ex

task="mmlu"
#task="mmlu_pro"
#task="USMLE"

#base_model="llama3.1"
#base_model="tinyllama1.1base"
#base_model="phi3mini4k"
#base_model="gemma2_2bit"
base_model="gemma2_2b"

### warm sft make data
#datatype=labeled
#input_file="./data/${task}/labeled.csv"
#output_file="./sft/data/${task}_labeled_alpaca.json"


### pseudo make data
datatype=pseudo_labeled
input_file="./data/${task}/pseudo_warm_${base_model}_${task}.csv"
output_file="./sft/data/pseudo_warm_${base_model}_${task}_alpaca.json"

## 测试
#datatype=pseudo_labeled
#input_file="./data/${task}/pseudo_warm_llama3.1_${task}.csv"
#### 可能需要手动改一下 conver_data.py的format_question_alpaca函数
#output_file="./sft/data/pseudo_warm_llama3.1_${task}_alpaca_test_pseudo.json"
#output_file="./sft/data/pseudo_warm_llama3.1_${task}_alpaca_test_all_right.json"
#output_file="./sft/data/pseudo_warm_llama3.1_${task}_alpaca_test_all_wrong.json"

python utils/convert_data.py $input_file $output_file $datatype