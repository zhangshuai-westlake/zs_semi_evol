import os
import re
import requests
import json

# url='http://bridge.xinchenai.com/v1/chat/completions'
# api_model_name = "text-embedding-3-small"
#
# data = {
#     "source": "joyland",
#     "model": api_model_name,
#     "encoding_format": "float",
#     "input": "just test",
# }
#
# response = requests.post(url, json=data, verify=False, timeout=500)
# print(response)
#
#
# import json
# path = "/data/users/zhangshuai/work/semi_instruction_tuning/SemiEvol/save/mmlu__data_users_zhangshuai_work_pretrained_models_Meta-Llama-3.1-8B-Instruct.json"
# answers = []
# with open(path, "r") as fo:
#     for line in fo.readlines():
#         answers.append(json.loads(line)["response"])


# import random
# import pandas as pd
# import os
# import sys
# sys.path.append('..')
# import openai
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
#
#
# os.environ['OPENAI_API_KEY'] = "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"
# embed_path = "tmp/USMLE_labeled"
# embed = OpenAIEmbeddings(model="text-embedding-3-small")
# vectorstore = FAISS.load_local(embed_path, embed, allow_dangerous_deserialization=True)
#
# print()


# import pandas as pd
# from tqdm import tqdm
# from common import format_multichoice_question
#
# data_path = "./data/USMLE/unlabeled.csv"
# df = pd.read_csv(data_path)
# examples = [row.to_dict() for _, row in df.iterrows()]
#
# instances = []
# print("start")
# for row in tqdm(examples):
#     instances.append(format_multichoice_question(row))
# print("end")



from vllm import LLM

model_path = "/storage/home/westlakeLab/zhangshuai/models/TinyLlama/TinyLlama_v1.1"

llm = LLM(model=model_path, task="generate")
output = llm.generate("Hello, my name is")
print(output)