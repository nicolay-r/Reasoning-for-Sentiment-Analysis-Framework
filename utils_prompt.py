# Similar to the following studies:
# Bowen Zhang, Daijun Ding, and Liwen Jing. 2022.
# How would stance detection techniques evolve after the
# launch of chatgpt? arXiv preprint arXiv:2212.14548.
common_prompts = {
    # This is a default prompt from which we initiate experiments with LLM application in task.
    # This prompt represent a copy from the paper: RuSentNE-2023 competition.
    "rusentrel2023_default_en": "What's the attitude of the sentence '{sentence}', " +
                                "to the target '{entity}'? " +
                                "Select one from: positive, negative, neutral.",
    "rusentrel2023_default_ru": "Какая оценка тональности в предложении '{sentence}', " +
                                "по отношению к '{entity}'? " +
                                "Выбери из трех вариантов: позитивная, негативная, нейтральная.",
    # Zero-shot CoT https://www.mercity.ai/blog-post/guide-to-chain-of-thought-prompting
    # Source: https://arxiv.org/abs/2205.11916
    "rusentrel2023_cot_en": "What's the attitude of the sentence '{sentence}', " +
                            "to the target '{entity}'? " +
                            "Let's think step by step." +
                            "Select one from: positive, negative, neutral.",
    "rusentrel2023_cot_ru": "Какая оценка тональности в предложении '{sentence}', " +
                            "по отношению к '{entity}'? " +
                            "Рассуждай по шагам."
                            "Выбери из трех вариантов: позитивная, негативная, нейтральная.",
    # This is a more-precise version of the prompt.
    "rusentrel2023_se_en": "What is the attitude of the author or another subject " +
                           "in the sentence '{sentence}' to the target '{entity}'?" +
                           "Choose from: positive, negative, neutral. ",
    "rusentrel2023_se_ru": "Каково отношение автора или другого субъекта" +
                           "в предложении '{sentence}' к '{entity}'?" +
                           "Выбери из трех вариантов: позитивная, негативная, нейтральная."
}

chatgpt_prompts = {
    "rusentrel2023_default_en_short": common_prompts["rusentrel2023_default_en"] +
                                      " Create a very short summary that uses 50 completion_tokens or less.",
    "rusentrel2023_default_ru_short": common_prompts["rusentrel2023_default_ru"] +
                                      " Составь короткий ответ из не более 50 completion_tokens.",
    # ------------------
    # 10/01/2024 Update:
    # ------------------
    # Experiments with other prompt modifications.
    # Purpose: investigate differences on results affection.
    # Result: These prompts won't affect the result performance on RuSentNE-2023.
    "rusentrel2023_default_sentiment_en_short": "What's the sentiment of the sentence '{sentence}', " +
                                                "to the target '{entity}'? " +
                                                "Select one from: positive, negative, neutral." +
                                                " Create a very short summary that uses 50 completion_tokens or less.",
    "rusentrel2023_default_opinion_en_short": "What's the author's opinion in the sentence '{sentence}', " +
                                              "to the target '{entity}'? " +
                                              "Select one from: positive, negative, neutral." +
                                              " Create a very short summary that uses 50 completion_tokens or less.",
    # ------------------
    # 12/01/2024.
    # ------------------
    # New version of the prompts.
    "rusentrel2023_se_ru_short":  common_prompts["rusentrel2023_se_ru"] +
                                  " Составь короткий ответ и пояснение, в котором будет использовано не более 50 completion_tokens",
    "rusentrel2023_se_en_short":  common_prompts["rusentrel2023_se_en"] +
                                  " Create a very short summary that uses 50 completion_tokens or less.",
    "rusentrel2023_es_en_short":  "What is the attitude of the author or another subject "
                                  "to the target '{entity}' in the sentence '{sentence}'?" +
                                  "Choose from: positive, negative, neutral." +
                                  " Create a very short summary that uses 50 completion_tokens or less.",
    # ------------------
    # 13/01/2024.
    # ------------------
    # The revised version in which we check whether
    # Answer first and Explain will allow to enhance the performance.
    "rusentrel2023_se_en_ae": common_prompts["rusentrel2023_se_en"] +
                              "Answer first and explain why.",
}

# Generated from the following prompt to ChatGPT-4
# > Using the ChatGPT4 for sentiment analysis, what whould be most suitable "assistant prompt" setup, like "you are ... "?
GPT4_SENTIMENT_ANALYSIS_ASSISTANT_PROMPT = \
    "You are an AI assistant skilled in natural language processing and sentiment analysis. " \
    "Your task is to analyze text inputs to determine the underlying sentiment, whether " \
    "it's positive, negative, or neutral. You should consider the nuances of language, " \
    "including sarcasm, irony, and context. Your responses should include not only the " \
    "sentiment classification but also a brief explanation of why a particular sentiment " \
    "was assigned, highlighting key words or phrases that influenced the decision."
