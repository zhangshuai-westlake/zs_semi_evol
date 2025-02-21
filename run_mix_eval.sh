set -ex

### base model
#model="llama3.1_base"
#model="tinyllama1.1_base"
#model="gemma2_2b"

### base instruction model
model="llama3.1"
#model="phi3mini4k"
#model="gemma2_2bit"

### warm sft model
#model="warm_${model}_mix"

########### semievol model
### baseline
#model="pseudo_${model}_mix"

port=6006

export CUDA_VISIBLE_DEVICES=0

python mix_eval.py $model $port