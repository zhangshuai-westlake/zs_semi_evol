### base model
#model_path="/backup/lanzhenzhongLab/public/shuai_share/Llama-3.1-8B"
#model_path="/storage/home/westlakeLab/zhangshuai/models/TinyLlama/TinyLlama_v1.1"
#model_path="/storage/home/westlakeLab/zhangshuai/models/google/gemma-2-2b"

### base instruction model
#model_path="/storage/home/westlakeLab/zhangshuai/models/Meta-Llama-3.1-8B-Instruct"
#model_path="/storage/home/westlakeLab/zhangshuai/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#model_path="/storage/home/westlakeLab/zhangshuai/models/microsoft/Phi-3-mini-4k-instruct"
#model_path="/storage/home/westlakeLab/zhangshuai/models/google/gemma-2-2b-it"


task="mmlu"
#task="USMLE"
#task="mmlu_pro"

#base_model="llama3.1"
#base_model="tinyllama1.1base"
#base_model="phi3mini4k"
#base_model="gemma2_2bit"
base_model="gemma2_2b"

### warm model
#model_path="./sft/output/merged_warm_${base_model}_${task}"

########### semievol model
### baseline
model_path="./sft/output/merged_pseudo_${base_model}_${task}_filter"
### test
#model_path="./sft/output/merged_pseudo_llama3.1_${task}_filter_test_pseudo"
#model_path="./sft/output/merged_pseudo_llama3.1_${task}_filter_test_all_right"
#model_path="./sft/output/merged_pseudo_llama3.1_${task}_filter_test_all_wrong"
#model_path="./sft/output/merged_pseudo_${base_model}_${task}_filter_test_for_template"
### threshold_by_filter_proportion
#model_path="./sft/output/merged_pseudo_${base_model}_${task}_filter_threshold_by_filter_proportion"



### paper report model
#model_path="/data/users/zhangshuai/work/pretrained_models/luojunyu/Llama-3.1-8B-SemiEvol-MMLU"

export CUDA_VISIBLE_DEVICES=6

vllm="/storage/home/westlakeLab/zhangshuai/anaconda3/envs/vllm/bin/vllm"
$vllm serve $model_path --port 6006

#chat_template="chat_template/template_gemma-it.jinja"
#$vllm serve $model_path --port 6006 --chat-template $chat_template
