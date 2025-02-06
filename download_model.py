# export HF_ENDPOINT=https://hf-mirror.com

from huggingface_hub import snapshot_download

# model_paths = ["luojunyu/Llama-3.1-8B-SemiEvol-MMLU"]
# model_paths = ["luojunyu/Llama-3.1-8B-SemiEvol-MMLUPro"]
# model_paths = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
# model_paths = ["microsoft/Phi-3-mini-4k-instruct"]
# model_paths = ["google/gemma-2-2b-it"]
model_paths = ["google/gemma-2-2b"]

for model_path in model_paths:
    print(f"Downloading {model_path}")
    local_dir = '/storage/home/westlakeLab/zhangshuai/models/' + model_path
    # local_dir = '/data/users/zhangshuai/work/pretrained_models/' + model_path
    snapshot_download(repo_id=model_path,
                      repo_type='model',
                      local_dir=local_dir,
                      resume_download=True,
                      token="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")