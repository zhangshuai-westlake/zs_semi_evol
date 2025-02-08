export CUDA_VISIBLE_DEVICES=5
#model_path="/backup/lanzhenzhongLab/public/shuai_share/Llama-3.1-8B"

#model_path="/storage/home/westlakeLab/zhangshuai/models/google/gemma-2-2b-it"
#model_path="/storage/home/westlakeLab/zhangshuai/models/google/gemma-2-2b"
#model_path="/storage/home/westlakeLab/zhangshuai/models/Meta-Llama-3.1-8B-Instruct"
model_path="/storage/home/westlakeLab/zhangshuai/models/microsoft/Phi-3-mini-4k-instruct"

#ds_path="openai/gsm8k"
#ds_split="test"
#ds_name="main"
#q_column_name="question"
#a_column_name="answer"

#ds_path="openai/openai_humaneval"
#ds_split="test"
#q_column_name="prompt"
#a_column_name="canonical_solution"

#ds_path="google-research-datasets/mbpp"
#ds_split="test"
#q_column_name="text"
#a_column_name="code"


#ds_path="qiaojin/PubMedQA"
#ds_name="pqa_labeled"
#ds_split="train"
#q_column_name="question"
#a_column_name="final_decision"
##a_column_name="long_answer"
#option_description="yes;no;maybe"


#ds_path="FinGPT/fingpt-convfinqa"
#ds_split="test"
#q_column_name="input"
#a_column_name="output"
#instruction_column_name="instruction"


#ds_path="cais/mmlu"
#ds_split="test"
##ds_name="abstract_algebra"
##ds_name="college_physics"
##ds_name="marketing"
##ds_name="professional_law"
#ds_name="sociology"
#q_column_name="question"
#a_column_name="answer"
#option_column_name="choices"


ds_path="TIGER-Lab/MMLU-Pro"
ds_split="test"
q_column_name="question"
a_column_name="answer"
option_column_name="options"



num_examples=100


python cal_ppl.py \
    --model_path $model_path \
    --ds_path $ds_path \
    --ds_split $ds_split \
    --num_examples $num_examples \
    --q_column_name $q_column_name \
    --a_column_name $a_column_name \
    --option_column_name $option_column_name \


#    --ds_name $ds_name \
#    --option_description $option_description \
#    --instruction_column_name $instruction_column_name \
#    --option_column_name $option_column_name \
