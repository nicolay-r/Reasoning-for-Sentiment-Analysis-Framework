# Reasoning for Sentiment Analysis

Studies and Collection of LLM-based Reasoning engines / frameworks for Sentiment Analysis.
This repository contains the code for the paper:
[Large Language Models in Targeted Sentiment Analysis for Russian]().

## Contents

* [Installation](#installation)
* [Preparing Data](#preparing-data)
    * [Manual Translation](#manual-data-translation)
* [**Zero-Shot**](#zero-shot)
    * [Examples](#usage-examples)
* [References](#references)

# Installation

```bash
pip install -r dependencies
```

# Preparing Data 

Simply launch the following script for obtaining both original texts and **Translated**:
```bash
python rusentne23_download.py
```

## Manual Data Translation
You could launch manual data translation to English language (`en`) via [GoogleTrans](https://github.com/ssut/py-googletrans):
```bash
 python rusentne23_translate.py --src data/train_data.csv --lang en --label
 python rusentne23_translate.py --src data/valid_data.csv --lang en --label
 python rusentne23_translate.py --src data/final_data.csv --lang en
 ```

# Zero-Shot

[[prompts]](utils_prompt.py)

This is a common script for launching LLM model inference in Zero-shot format using manual or 
[predefined prompts](utils_prompt.py):

```python
python zero_shot_infer.py \
        --model "google/flan-t5-base" \
        --src "data/final_data_en.csv" \
        --prompt "rusentne2023_default_en" \
        --device "cpu" \
        --to "csv" \
        --temp 0.1 \
        --output "data/output.csv" \
        --max-length 512 \
        --hf-token "<YOUR_HUGGINGFACE_TOKEN>" \
        --openai-token "<YOUR_OPENAI_TOKEN>" \
        --limit 10000 \
        --limit-prompt 10000 \
        --bf16 \
        --l4b
```

### Notes

* **Mixtral-7B**: [you need to install `transformers 4.36.0`, read the blog post](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/discussions/9); 
  [[model size]](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/discussions/3#657721f02eb103d91fd044f1)


<details>
<summary>

### Usage Examples

</summary>

#### Chat mode

Simply setup `model` name and `device` you wish to use for launching model.

```bash
python zero_shot_infer.py --model google/flan-t5-base --device cpu
```

#### Inference with the predefined prompt 

Use the `prompt` command for passing the [predefined prompt](utils_prompt.py) or textual prompt that involves the `{text}` information. 

```bash
python zero_shot_infer.py --model google/flan-t5-small \
    --device cpu --src data/final_data_en.csv --prompt 'rusentrel2023_default_en'
```

#### Zero-Shot CoT 
[![arXiv](https://img.shields.io/badge/arXiv-2205.11916-b31b1b.svg)](https://arxiv.org/abs/2205.11916)

```bash
python zero_shot_infer.py --model mistralai/Mistral-7B-Instruct-v0.1 --src data/final_data_en.csv \
    --prompt "What's the attitude of the sentence '{sentence}' to the target '{entity}'? Let's think step by step." --output "zeroshot-cot-hop1-{model}.csv"
python zero_shot_infer.py --model mistralai/Mistral-7B-Instruct-v0.1 --src "zeroshot-cot-hop1-{model}.csv" \
    --prompt "{prompt}{response}. Therefore the sentiment class (positive, negative, neutral) is"
```

#### OpenAI models

Use the `model` parameter prefixed by `openai:`, followed by 
[model names](https://github.com/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework/blob/b8e588e4722c27c88acd33bbaeabeee00a903688/zero_shot_infer.py#L63-L79) 
as follows:

```bash
python zero_shot_infer.py --model "openai:gpt-3.5-turbo-1106" \
    --src "data/final_data_en.csv" --prompt "rusentrel2023_default_en_short" \
    --max-length 75 --limit 5
```

</details>


## References

> TODO.