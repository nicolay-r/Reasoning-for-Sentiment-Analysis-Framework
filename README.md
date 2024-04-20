# Reasoning for Sentiment Analysis
![](https://img.shields.io/badge/Python-3.8-brightgreen.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework/blob/main/Reasoning_for_Sentiment_Analysis_Framework.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2404.12342-b31b1b.svg)](https://arxiv.org/abs/2404.12342)

> **💻 Update 19/04/2024:** We open [quick_cot](https://github.com/nicolay-r/quick_cot) code repository for lauching quick CoT zero-shot-learning / few-shot-learning experiments with LLM, utilied in this studies. [More ...](https://github.com/nicolay-r/quick_cot)

> **📊 Update 19/04/2024:** We open a separate 📊 👉[RuSentNE-benchmark repository](https://github.com/nicolay-r/RuSentNE-LLM-Benchmark)👈 📊 for LLM-resonses, including **answers on reasoning steps in THoR CoT** for ChatGPT model series.
> [More ...](https://github.com/nicolay-r/RuSentNE-LLM-Benchmark)


Studies and Collection of LLM-based reasoning frameworks for Target Sentiment Analysis.
This repository contains source code for paper @ [LJoM journal](https://link.springer.com/journal/12202) titled as:
[Large Language Models in Targeted Sentiment Analysis for Russian](https://arxiv.org/abs/2404.12342).


## Contents

* [Installation](#installation)
* [Preparing Data](#preparing-data)
    * [Manual Translation](#manual-data-translation)
* [**Zero-Shot**](#zero-shot)
    * [Examples](#usage-examples)
* [**Chain-of-Thought fine-tuning**](#three-hop-chain-of-thought-thor)
* [**Answers**](#answers)
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

```bash
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

# Zero-Shot Chain-of-Thought 

This functionality if out-of-scope of this repository.

We release a tiny framework, dubbed as [quick_cot](https://github.com/nicolay-r/quick_cot) for applying CoT schemas, with API similar to one in  [**Zero-Shot**](#zero-shot) section, based on **schemas** written in JSON notation.

### 📝 👉 [`thor-zero-shot-cot-english-shema.json`](config/thor-zero-shot-cot-english-schema.json) 👈
### 💻 👉 [Tiny CoT-framework (quick_cot)](https://github.com/nicolay-r/quick_cot) 👈

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

## Answers

Results of the zero-shot models obtained during experiments fall outside the scope of this repository.
We open a separate  for LLM-resonses, including **answers on reasoning steps in THoR CoT** for ChatGPT model series:

### 👉 [RuSentNE-benchmark repository](https://github.com/nicolay-r/RuSentNE-LLM-Benchmark) 👈

## References

You can cite this work as follows:
```bibtex
@misc{rusnachenko2024large,
      title={Large Language Models in Targeted Sentiment Analysis}, 
      author={Nicolay Rusnachenko and Anton Golubev and Natalia Loukachevitch},
      year={2024},
      eprint={2404.12342},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
