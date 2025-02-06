
#task="USMLE"
task="mmlu"
#task="mmlu_pro"

#base_model="llama3.1"
#base_model="tinyllama1.1base"
base_model="phi3mini4k"

path="data/${task}/pseudo_warm_${base_model}_${task}.csv"

python cal_filter_data_acc.py $path