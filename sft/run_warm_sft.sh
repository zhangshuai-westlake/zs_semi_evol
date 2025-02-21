#conda activate llama_factory
export CUDA_VISIBLE_DEVICES=0,1,2,3
llamafactory-cli train warm_sft.yaml