set -ex

### 1.embedding过程
conda activate semi_instruction_tuning
python mix_retrieval.py

### 2.self-training过程
conda activate semi_instruction_tuning
sh run_mix_convert_data.sh
cd sft
conda activate llama_factory
sh run_warm_sft.sh   # 修改warm_sft.yaml(可能要修改template), dataset_info.json
sh run_warm_merge.sh  # 修改warm_merge.yaml(可能要修改template)

cd ..
conda activate vllm
sh run_vllm.sh
conda activate semi_instruction_tuning
sh run_mix_semievol.sh  # 修改config.py

sh run_mix_convert_data.sh
#修改伪标签筛选阈值的方法后生成数据，从这里开始
cd sft
conda activate llama_factory
sh run_sft.sh  # 修改sft.yaml(可能要修改template), dataset_info.json
sh run_merge.sh # 修改merge.yaml(可能要修改template)

### 3.评估
cd ..
conda activate vllm
sh run_vllm.sh
conda activate semi_instruction_tuning
sh run_mix_eval.sh  # 修改config.py


### 4.计算中间过程
sh run_cal_filter_data_acc.sh
sh run_cal_acc.sh

