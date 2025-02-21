ZS_MULTICHOICE_TEMPLATE = """
Answer the multiple choice question. Your response should be of the following format: 'Answer: LETTER' (without quotes).

Question: 
{question}
Options:
{options}
""".strip()
QA_MULTICHOICE_TEMPLATE = """
Question: 
{question}
Options:
{options}
{answer}
""".strip()
FS_MULTICHOICE_TEMPLATE = """
Answer the multiple choice question. Your response should be of the following format: 'Answer: LETTER' (without quotes).
Below are some examples.
{reference}
Here is the new task.
Question: 
{question}
Options:
{options}
""".strip()


ZS_GSM8K_TEMPLATE = """
Answer the following math question. Provide the solution and final answer in the format: `{{SOLUTION}}\n#### {{ANSWER}}` (without quotes). The variable `SOLUTION` should include the detailed steps, and the variable `ANSWER` should be the final answer in digits only. Ensure there are exactly four `#` i.e. `####` between `SOLUTION` and `ANSWER`.

Question: 
{question}
Answer:
""".lstrip()
QA_GSM8K_TEMPLATE = """
Question: 
{question}
Answer:
{answer}
""".strip()
FS_GSM8K_TEMPLATE = """
Answer the following math question. Provide the solution and final answer in the format: `{{SOLUTION}}\n#### {{ANSWER}}` (without quotes). The variable `SOLUTION` should include the detailed steps, and the variable `ANSWER` should be the final answer in digits only. Ensure there are exactly four `#` i.e. `####` between `SOLUTION` and `ANSWER`.

Below are some examples.
{reference}
Here is the new task.
Question: 
{question}
Answer:
""".lstrip()


# ZS_OPENAI_HUMANEVAL_TEMPLATE = """
# Complete the following code. Your response should only the added code without any additional examples, demonstrations, or explanations.
#
# {question}
#
# """.strip()
ZS_OPENAI_HUMANEVAL_TEMPLATE = """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Do not provide additional examples, demonstrations, or explanations. Here is the given code to do completion:

Question:
{question}
Answer:
""".lstrip()
QA_OPENAI_HUMANEVAL_TEMPLATE = """
Question: 
{question}
Answer:
{answer}
""".strip()
FS_OPENAI_HUMANEVAL_TEMPLATE = """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Do not provide additional examples, demonstrations, or explanations. Here is the given code to do completion:

Below are some examples.
{reference}
Here is the new task.
Question:
{question}
Answer:
""".lstrip()


ZS_CONVFINQA_TEMPLATE = """
{instruction}
Your response should be of the following format: 'Answer: VALUE' (without quotes, VALUE should be digits).

Question: 
{question}
""".strip()
QA_CONVFINQA_TEMPLATE = """
Question: 
{question}{answer}
""".strip()
FS_CONVFINQA_TEMPLATE = """
{instruction}
Your response should be of the following format: 'Answer: VALUE' (without quotes, VALUE should be digits).

Below are some examples.
{reference}
Here is the new task.
Question: 
{question}
""".strip()


ZS_MBPP_TEMPLATE = """
Please generate a python function for my problem. Your response should only the executable code of the function without any additional examples, demonstrations, or explanations.
>>> Problem: 
{question}
>>> Test Cases:
{tests}
""".strip()
QA_MBPP_TEMPLATE = """
>>> Problem: 
{question}
>>> Test Cases:
{tests}
{answer}
""".strip()
FS_MBPP_TEMPLATE = """
Please generate a python function for my problem. Your response should only the executable code of the function without any additional examples, demonstrations, or explanations.

Below are some examples.
{reference}
Here is the new task.
>>> Problem: 
{question}
>>> Test Cases:
{tests}
""".strip()


ZS_PRM800K_TEMPLATE = """
Answer the following math question. Provide the solution and final answer in the format: `{{SOLUTION}}\n#### {{ANSWER}}` (without quotes). The variable `SOLUTION` should include the detailed steps, and the variable `ANSWER` should be the final answer. Ensure there are exactly four `#` i.e. `####` between `SOLUTION` and `ANSWER`.

Question: 
{question}
Answer:
""".lstrip()
QA_PRM800K_TEMPLATE = """
Question: 
{question}
Answer:
{answer}
""".strip()
FS_PRM800K_TEMPLATE = """
Answer the following math question. Provide the solution and final answer in the format: `{{SOLUTION}}\n#### {{ANSWER}}` (without quotes). The variable `SOLUTION` should include the detailed steps, and the variable `ANSWER` should be the final answer. Ensure there are exactly four `#` i.e. `####` between `SOLUTION` and `ANSWER`.

Below are some examples.
{reference}
Here is the new task.
Question: 
{question}
Answer:
""".lstrip()