from common import *

MODELS_CONFIG = {
    ### base instruction model
    "llama3.1": {
        # "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        # "name": "/data/users/zhangshuai/work/pretrained_models/Meta-Llama-3.1-8B-Instruct",
        "name": "/storage/home/westlakeLab/zhangshuai/models/Meta-Llama-3.1-8B-Instruct",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "tinyllama1.1": {
        "name": "/storage/home/westlakeLab/zhangshuai/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "phi3mini4k": {
        "name": "/storage/home/westlakeLab/zhangshuai/models/microsoft/Phi-3-mini-4k-instruct",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "gemma2_2bit": {
        "name": "/storage/home/westlakeLab/zhangshuai/models/google/gemma-2-2b-it",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },

    ### base model
    "llama3.1_base": {
        "name": "/backup/lanzhenzhongLab/public/shuai_share/Llama-3.1-8B",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "tinyllama1.1_base": {
        "name": "/storage/home/westlakeLab/zhangshuai/models/TinyLlama/TinyLlama_v1.1",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "gemma2_2b": {
        "name": "/storage/home/westlakeLab/zhangshuai/models/google/gemma-2-2b",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },

    ### warm sft model
    #llama3.1
    "warm_llama3.1_mmlu": {
        "name": "./sft/output/merged_warm_llama3.1_mmlu",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "warm_llama3.1_USMLE": {
        "name": "./sft/output/merged_warm_llama3.1_USMLE",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "warm_llama3.1_mmlu_pro": {
        "name": "./sft/output/merged_warm_llama3.1_mmlu_pro",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    #tinyllama1.1base
    "warm_tinyllama1.1base_mmlu": {
        "name": "./sft/output/merged_warm_tinyllama1.1base_mmlu",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    #phi3mini4k
    "warm_phi3mini4k_mmlu": {
        "name": "./sft/output/merged_warm_phi3mini4k_mmlu",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    #gemma2_2bit
    "warm_gemma2_2bit_mmlu": {
        "name": "./sft/output/merged_warm_gemma2_2bit_mmlu",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    #gemma2_2b
    "warm_gemma2_2b_mmlu": {
        "name": "./sft/output/merged_warm_gemma2_2b_mmlu",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },

    ################ semievol model
    ###baseline
    "pseudo_llama3.1_mmlu": {
        "name": "./sft/output/merged_pseudo_llama3.1_mmlu_filter",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "pseudo_llama3.1_USMLE": {
        "name": "./sft/output/merged_pseudo_llama3.1_USMLE_filter",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "pseudo_llama3.1_mmlu_pro": {
        "name": "./sft/output/merged_pseudo_llama3.1_mmlu_pro_filter",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "pseudo_tinyllama1.1base_mmlu": {
        "name": "./sft/output/merged_pseudo_tinyllama1.1base_mmlu_filter",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "pseudo_phi3mini4k_mmlu":{
        "name": "./sft/output/merged_pseudo_phi3mini4k_mmlu_filter",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "pseudo_gemma2_2b_mmlu": {
        "name": "./sft/output/merged_pseudo_gemma2_2b_mmlu_filter",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },

    ### test
    "pseudo_llama3.1_mmlu_pro_test_pseudo": {
        "name": "./sft/output/merged_pseudo_llama3.1_mmlu_pro_filter_test_pseudo",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "pseudo_llama3.1_mmlu_pro_test_all_right": {
        "name": "./sft/output/merged_pseudo_llama3.1_mmlu_pro_filter_test_all_right",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "pseudo_llama3.1_mmlu_pro_test_all_wrong": {
        "name": "./sft/output/merged_pseudo_llama3.1_mmlu_pro_filter_test_all_wrong",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "pseudo_tinyllama1.1base_mmlu_test_for_template": {
        "name": "./sft/output/merged_pseudo_tinyllama1.1base_mmlu_filter_test_for_template",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },

    ### threshold_by_filter_proportion
    "pseudo_llama3.1_mmlu_threshold_by_filter_proportion": {
        "name": "./sft/output/merged_pseudo_llama3.1_mmlu_filter_threshold_by_filter_proportion",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "pseudo_llama3.1_mmlu_pro_threshold_by_filter_proportion": {
        "name": "./sft/output/merged_pseudo_llama3.1_mmlu_pro_filter_threshold_by_filter_proportion",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "pseudo_phi3mini4k_mmlu_threshold_by_filter_proportion": {
        "name": "./sft/output/merged_pseudo_phi3mini4k_mmlu_filter_threshold_by_filter_proportion",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },

    ### paper report semievol model
    "llama3.1-mmlu-labeled": {
        # "name": "mmlu_labeled",
        "name": "/data/users/zhangshuai/work/pretrained_models/luojunyu/Llama-3.1-8B-SemiEvol-MMLU",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
     "llama3.1-mmlu_pro_labeled": {
        "name": "mmlu_pro_labeled",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "llama3.1-mmlu_pro_pseudo": {
        "name": "mmlu_pro_pseudo",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "llama3.1-arc_labeled": {
        "name": "arc_labeled",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "llama3.1-arc_pseudo": {
        "name": "arc_pseudo",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "Reflection-Llama-3.1-8B": {
        "name": "Solshine/reflection-llama-3.1-8B",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "Hermes-3-Llama-3.1-8B": {
        "name": "NousResearch/Hermes-3-Llama-3.1-8B",
        "url": "http://localhost:6003/v1",
        "method": "loop"
    },
    "adaptllm-med": {
        "name": "AdaptLLM/medicine-chat",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "adaptllm-fin": {
        "name": "AdaptLLM/finance-chat",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "instuctpt-med": {
        "name": "AdaptLLM/instuctpt-chat",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "instuctpt-fin": {
        "name": "AdaptLLM/instuctpt-chat",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "memoryllm": {
        "name": "memoryllm",
        "url": "http://localhost:6006/v1",
        "method": "loop"
    },
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    },
    "gpt-4o-mini-arc_labeled": {
        "name": "ft:gpt-4o-mini-2024-07-18:xxxx",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    },
    "gpt-4o-mini-usmle_labeled": {
        "name": "ft:gpt-4o-mini-2024-07-18:xxxx",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    },
    "gpt-4o-mini-pubmedqa_labeled": {
        "name": "ft:gpt-4o-mini-2024-07-18:xxxx",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    },
    "gpt-4o-mini-fpb_labeled": {
        "name": "ft:gpt-4o-mini-2024-07-18:xxxx",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    },
    "gpt-4o-mini-convfinqa_labeled": {
        "name": "ft:gpt-4o-mini-2024-07-18:xxxx",
        "url": "https://api.openai.com/v1",
        "method": "batch"
    }
}


TASK_CONFIG = {
    'mmlu': 'multiple_choice',
    'mmlu_pro' :'multiple_choice',
    'arc':'multiple_choice',
    'FPB' :'multiple_choice',
    'PubMedQA':'multiple_choice',
    'USMLE':'multiple_choice',
    'ConvFinQA' :'math'
}

FUNCTION_UTILS = {
    'math': {
        'format_fn': format_value_prompt,
        'check_fn': check_answer,
        'few_shot_prompt': FEW_SHOT_VALUE_SYSTEM,
        'reflection_fn': format_reflection_value,
    },
    'multiple_choice': {
        'format_fn': format_multichoice_question,
        'check_fn': check_answer,
        'few_shot_prompt': FEW_SHOT_SYSTEM,
        'reflection_fn': format_reflection,
    }
}

EVAL_UTILS = {
    'math': {
        'format_fn': format_value_question,
        'check_fn': check_answer_value,
        'extract_fn': extract_value,
    },
    'multiple_choice': {
        'format_fn': format_question,
        'check_fn': check_answer,
        'extract_fn': extract_result,
    }
}
