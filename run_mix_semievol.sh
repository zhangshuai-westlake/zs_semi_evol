base_model="llama3.1"
#base_model="tinyllama1.1base"
#base_model="phi3mini4k"
#base_model="gemma2_2b"

#model=$base_model
model="warm_${base_model}_mix"

export CUDA_VISIBLE_DEVICES=0


port=6006

python mix_semievol.py $model $port
