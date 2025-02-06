task="mmlu"
#task="USMLE"
#task="mmlu_pro"

#base_model="llama3.1"
#base_model="tinyllama1.1base"
#base_model="phi3mini4k"
#base_model="gemma2_2bit"
base_model="gemma2_2b"

### base model
#name=mmlu__storage_home_westlakeLab_zhangshuai_models_TinyLlama_TinyLlama_v1.1

### base instruction model
#name=${task}__storage_home_westlakeLab_zhangshuai_models_Meta-Llama-3.1-8B-Instruct
#name=${task}__storage_home_westlakeLab_zhangshuai_models_google_gemma-2-2b-it

#### warm model
name=${task}_._sft_output_merged_warm_${base_model}_${task}
#
#### semievol model
#name=${task}_._sft_output_merged_pseudo_${base_model}_${task}_filter
#name=${task}_._sft_output_merged_pseudo_${base_model}_${task}_filter_threshold_by_filter_proportion

data_path="./data/${task}/test.csv"
rest_path="./save/${name}.json"

python cal_acc.py $data_path $rest_path