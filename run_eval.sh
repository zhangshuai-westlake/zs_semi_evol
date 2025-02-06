set -ex

task="mmlu"
#task="USMLE"
#task="mmlu_pro"

### base model
#model="llama3.1_base"
#model="tinyllama1.1_base"
#model="gemma2_2b"

### base instruction model
#model="llama3.1"
#model="phi3mini4k"
#model="gemma2_2bit"


#model_name="llama3.1_base"
#model_name="tinyllama1.1_base"
#model_name="llama3.1"
#model_name="phi3mini4k"
#model_name="gemma2_2bit"
model_name="gemma2_2b"


 ### warm sft model
#model="warm_${model_name}_${task}"

########### semievol model
### baseline
model="pseudo_${model_name}_${task}"
### test
#model="pseudo_llama3.1_${task}_test_pseudo"
#model="pseudo_llama3.1_${task}_test_all_right"
#model="pseudo_llama3.1_${task}_test_all_wrong"
#model="pseudo_tinyllama1.1base_${task}_test_for_template"
### threshold_by_filter_proportion
#model="pseudo_${model_name}_${task}_threshold_by_filter_proportion"

python eval.py $task $model