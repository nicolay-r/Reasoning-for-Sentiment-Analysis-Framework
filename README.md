# Reasoning for Sentiment Analysis
![](https://img.shields.io/badge/Python-3.8-brightgreen.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework/blob/main/Reasoning_for_Sentiment_Analysis_Framework.ipynb)

Studies and Collection of LLM-based Reasoning engines / frameworks for Sentiment Analysis.
This repository contains the code for the paper:
[Large Language Models in Targeted Sentiment Analysis for Russian]().

## Contents

* [Installation](#installation)
* [Preparing Data](#preparing-data)
    * [Manual Translation](#manual-data-translation)
* [**Zero-Shot**](#zero-shot)
    * [Examples](#usage-examples)
* [**Chain-of-Thought fine-tuning**](#three-hop-chain-of-thought-thor)
* [References](#references)

## Installation

We separate dependencies necessary for zero-shot and fine-tuning experiments:
```bash
pip install -r dependencies_zs.txt
pip install -r dependencies_ft.txt
```

## Preparing Data 

Simply launch the following script for obtaining both original texts and **Translated**:
```bash
python rusentne23_download.py
```

## Manual Data Translation
You could launch manual data translation to English language (`en`) via [GoogleTrans](https://github.com/ssut/py-googletrans):
```bash
 python rusentne23_translate.py --src "data/train_data.csv" --lang "en" --label
 python rusentne23_translate.py --src "data/valid_data.csv" --lang "en" --label
 python rusentne23_translate.py --src "data/final_data.csv" --lang "en"
 ```

# Zero-Shot
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework/blob/main/Reasoning_for_Sentiment_Analysis_Framework.ipynb)
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

The application of Two-hop chain-of-thought concept for `Mistral-7B` model:

```bash
# Step 1.
python zero_shot_infer.py --model mistralai/Mistral-7B-Instruct-v0.1 \
    --src data/final_data_en.csv \
    --prompt "What's the attitude of the sentence '{sentence}' to the target '{entity}'? Let's think step by step." \
    --output "zeroshot-cot-hop1-{model}.csv"
# Step 2.
python zero_shot_infer.py --model mistralai/Mistral-7B-Instruct-v0.1 \ 
    --src "zeroshot-cot-hop1-{model}.csv" \
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

# Three Hop Chain-of-Thought THoR  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework/blob/main/Reasoning_for_Sentiment_Analysis_Framework.ipynb)

```bash
python thor_finetune.py -r "thor" -d "rusentne2023" 
    -li <PRETRAINED_STATE_INDEX> \
    -bs <BATCH_SIZE> \
    -es <EPOCH_SIZE> \
    -f "./config/config.yaml" 
```

<details>
<summary>

### Parameters list
</summary>

* `-c`, `--cuda_index`: Index of the GPU to use for computation (default: `0`).
* `-d`, `--data_name`: Name of the dataset (`rusentne2023`)
* `-r`, `--reasoning`: Specifies the reasoning mode (engine), with single `prompt` or multi-step `thor` mode.
* `-li`, `--load_iter`: load a state on specific index from the same `data_name` resource (default: `-1`, not applicable.)
* `-es`, `--epoch_size`: amount of training epochs (default: `1`)
* `-bs`, `--batch_size`: size of the batch (default: `None`)
* `-t`, `--temperature`: temperature (default=gen_config.temperature)
* `-z`, `--zero_shot`: running zero-shot inference with chosen engine on `test` dataset to form answers.
* `-f`, `--config`: Specifies the location of [config.yaml](config/config.yaml) file.

Configure more parameters in [config.yaml](config/config.yaml) file.

</details>


## References

> TODO.
