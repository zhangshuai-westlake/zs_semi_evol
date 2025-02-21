server_user="zhangshuai"
server_ip="172.16.75.200"

#server_dir="/storage/home/westlakeLab/zhangshuai/work/semi_instruction_tuning/SemiEvol/jupyterbook"
#local_dir="jupyterbook_backup"

#server_dir="/storage/home/westlakeLab/zhangshuai/work/semi_instruction_tuning/SemiEvol/achive/mmlu/ideal_balance_filter/jupyterbook_result"
#local_dir="analysis_result/mmlu/ideal_balance_filter"
#server_dir="/storage/home/westlakeLab/zhangshuai/work/semi_instruction_tuning/SemiEvol/achive/mmlu/soft_balance_filter/jupyterbook_result"
#local_dir="analysis_result/mmlu/soft_balance_filter"
#server_dir="/storage/home/westlakeLab/zhangshuai/work/semi_instruction_tuning/SemiEvol/achive/mmlu/ideal_label/jupyterbook_result"
#local_dir="analysis_result/mmlu/ideal_label"

#server_dir="/storage/home/westlakeLab/zhangshuai/work/semi_instruction_tuning/SemiEvol/achive/mmlu_pro/ideal_balance_filter/jupyterbook_result"
#local_dir="analysis_result/mmlu_pro/ideal_balance_filter"
#server_dir="/storage/home/westlakeLab/zhangshuai/work/semi_instruction_tuning/SemiEvol/achive/mmlu_pro/soft_balance_filter/jupyterbook_result"
#local_dir="analysis_result/mmlu_pro/soft_balance_filter"
#server_dir="/storage/home/westlakeLab/zhangshuai/work/semi_instruction_tuning/SemiEvol/achive/mmlu_pro/ideal_label/jupyterbook_result"
#local_dir="analysis_result/mmlu_pro/ideal_label"
#server_dir="/storage/home/westlakeLab/zhangshuai/work/semi_instruction_tuning/SemiEvol/achive/mmlu_pro_0.25label/ideal_balance_filter/jupyterbook_result"
#local_dir="analysis_result/mmlu_pro_0.25label/ideal_balance_filter"
#server_dir="/storage/home/westlakeLab/zhangshuai/work/semi_instruction_tuning/SemiEvol/achive/mmlu_pro_0.25label/soft_balance_filter/jupyterbook_result"
#local_dir="analysis_result/mmlu_pro_0.25label/soft_balance_filter"

server_dir="/storage/home/westlakeLab/zhangshuai/work/semi_instruction_tuning/SemiEvol/data/mix"
local_dir="remote_data"

scp -r $server_user@$server_ip:$server_dir ./$local_dir

# 存储服务器jupyter notebook文件到本地
#files=("analysis.py")
#files=("analysis.ipynb")

#files=("mmlu_test.csv" "mmlu_unlabeled.csv" "mmlu_compare.csv")
#files=("mmlu_pro_test.csv" "mmlu_pro_unlabeled.csv" "mmlu_pro_test_compare.csv")


#function sync(){
#  fpath=$1
#  fname=$2
#  echo "scp -r $server_user@$server_ip:$fpath/$fname ./$local_dir"
#  scp -r $server_user@$server_ip:$fpath/$fname ./$local_dir
#}
#
#echo "sync files"
#
#for file in $files
#do
#  echo "sync $file"
#  echo $file
#  sync $server_dir $file
#done