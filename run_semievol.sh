task="mmlu"
#task="USMLE"
#task="mmlu_pro"

#base_model="llama3.1"
#base_model="tinyllama1.1base"
#base_model="phi3mini4k"
base_model="gemma2_2b"


model="warm_${base_model}_${task}"


python semievol.py $task $model   # standand semievol

#topk=1
#python semievol.py $task $model $topk  # topk semievol to avoid too long input
