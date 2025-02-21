export CUDA_VISIBLE_DEVICES=5
#model_path="/backup/lanzhenzhongLab/public/shuai_share/Llama-3.1-8B"

#model_path="/storage/home/westlakeLab/zhangshuai/models/google/gemma-2-2b-it"
#model_port=6006

model_path="/storage/home/westlakeLab/zhangshuai/models/Meta-Llama-3.1-8B-Instruct"
model_port=6007

#model_path="/storage/home/westlakeLab/zhangshuai/models/microsoft/Phi-3-mini-4k-instruct"
#model_port=6008


ds_path="qiaojin/PubMedQA"
ds_name="pqa_labeled"
ds_split="train"
q_column_name="question"
a_column_name="final_decision"
#a_column_name="long_answer"
options="yes|no|maybe"


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


#ds_path="TIGER-Lab/MMLU-Pro"
#ds_split="test"
#q_column_name="question"
#a_column_name="answer"
#option_column_name="options"


num_examples=100

temperature=0.0

python cal_instruction_model_ppl.py \
    --model_path $model_path \
    --model_port $model_port \
    --ds_path $ds_path \
    --ds_split $ds_split \
    --num_examples $num_examples \
    --q_column_name $q_column_name \
    --a_column_name $a_column_name \
    --temperature $temperature \
    --ds_name $ds_name \
    --options $options \


#    --ds_name $ds_name \
#    --options $options \
#    --option_column_name $option_column_name \
