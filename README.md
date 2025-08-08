# Reasoning for Sentiment Analysis • [![twitter](https://img.shields.io/twitter/url/https/shields.io.svg?style=social)](https://twitter.com/nicolayr_/status/1781330684289658933)
![](https://img.shields.io/badge/Python-3.10-brightgreen.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework/blob/main/Reasoning_for_Sentiment_Analysis_Framework.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2404.12342-b31b1b.svg)](https://arxiv.org/abs/2404.12342)

This repository represent studies and Collection of **LLM-powered reasoning frameworks** 🧠 for **Target Sentiment Analysis**.
It contains source code for paper @ [LJoM journal](https://link.springer.com/journal/12202) titled as:
[Large Language Models in Targeted Sentiment Analysis for Russian](https://arxiv.org/abs/2404.12342).

<div align="center">

[![](https://markdown-videos-api.jorgenkh.no/youtube/qawLJsRHzB4)](https://youtu.be/qawLJsRHzB4)
[![](https://markdown-videos-api.jorgenkh.no/youtube/kdkZsTQibm0)](https://youtu.be/kdkZsTQibm0)

</div> 


<details>
<summary>

### **Updates**   

> **Update February 23 2025:** 🔥 Batching mode support. See 🌌 [Flan-T5 provider](https://github.com/nicolay-r/nlp-thirdgate/blob/master/llm/transformers_flan_t5.py) for [bulk-chain](https://github.com/nicolay-r/bulk-chain) project. Test [is available here](https://github.com/nicolay-r/bulk-chain/blob/master/test/test_provider_batching.py)

> **Update November 01 2024:** ⭐ Implemented a separated [bulk-chain](https://github.com/nicolay-r/bulk-chain) project for handling massive amount of prompts with CoT. This concept was used in this studies.

...

</summary>

> **Update 06 September 2024:** Mentioning the related information about the project at [BU-research-blog](https://blogs.bournemouth.ac.uk/research/2024/09/06/presenting-studies-on-llms-reasoning-capabilities-in-sentiment-analysis-of-mass-media-texts-at-nlpsummit-2024/)

> **Update 11 August 2024:** 🎤 Announcing the talk on this framework @ [NLPSummit 2024](https://www.nlpsummit.org/nlp-summit-2024/) with the preliminary ad and details in [X/Twitter post](https://x.com/JohnSnowLabs/status/1821999936478572836) 🐦. [![twitter](https://img.shields.io/twitter/url/https/shields.io.svg)](https://x.com/JohnSnowLabs/status/1821999936478572836)

> **Update 05 June 2024:** The frameworkless lauch of the **THoR-tuned FlanT5 model** for direct inference is now available in [GoogleColab](https://github.com/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework/blob/main/FlanT5_Finetuned_Model_Usage.ipynb).

> **Update 02 June 2024:** 🤗 Models were uploaded on `huggingface`: [🤗 nicolay-r/flan-t5-tsa-prompt-xl](https://huggingface.co/nicolay-r/flan-t5-tsa-thor-xl) and other models smaller sizes as well. Check out the [related section](#fine-tuned-flan-t5).

> **Update 22 May 2024:** ⚠️ in `GoogleColab` you might find zero evaluation results (see [#8 issue](https://github.com/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework/issues/8)) (`0`) on `test` part, which is due to locked availablity on labels 🔐. In order to apply your model for `test` part, please proceed with the [Official RuSentNE-2023 codalab competition page](https://codalab.lisn.upsaclay.fr/competitions/9538) or at [Github](https://github.com/dialogue-evaluation/RuSentNE-evaluation).

> **Update 06 March 2024**: 🔓 `attrdict` represents the main limitation for code launching in `Python 3.10` and hence been switched to `addict` (see [Issue#7](https://github.com/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework/issues/7)).

> **🔥 Update 24/04/2024:** We released fine-tuning log for the [prompt-based](DOC_FLANT5_PROMPT_FT.md) and [THoR-based](DOC_FLANT5_THOR_FT.md) techniques applied for the `train` competition data as well as **[checkpoints](#fine-tuned-flan-t5)** for downloading. [More ...](#fine-tuned-flan-t5)

> **💻 Update 19/04/2024:** We open [quick_cot](https://github.com/nicolay-r/quick_cot) code repository for lauching quick CoT zero-shot-learning / few-shot-learning experiments with LLM, utilied in this studies. [More ...](https://github.com/nicolay-r/quick_cot)

> **📊 Update 19/04/2024:** We open a separate 📊 👉[RuSentNE-benchmark repository](https://github.com/nicolay-r/RuSentNE-LLM-Benchmark)👈 📊 for LLM-resonses, including **answers on reasoning steps in THoR CoT** for ChatGPT model series.
> [More ...](https://github.com/nicolay-r/RuSentNE-LLM-Benchmark)

</details>


## Contents

* [Installation](#installation)
* [Preparing Data](#preparing-data)
    * [Manual Translation](#manual-data-translation)
* [**Zero-Shot**](#zero-shot)
    * [Examples](#usage-examples)
* [**Chain-of-Thought fine-tuning**](#three-hop-chain-of-thought-thor)
* [**Fine-tuned Flan-T5 Checkpoints 🔥**](#fine-tuned-flan-t5)
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
[![](https://img.shields.io/badge/prompts-ffffff.svg)](utils_prompt.py)


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

# Fine-tuned Flan-T5

👉 [**Prompt-Fine-Tuning** Logs](DOC_FLANT5_PROMPT_FT.md)

👉 [**THoR-Fine-Tuning** Logs](DOC_FLANT5_THOR_FT.md)


|Model|prompt|THoR|
|-----|------|----|
|**FlanT5-base** | - |[🤗 nicolay-r/flan-t5-tsa-thor-base](https://huggingface.co/nicolay-r/flan-t5-tsa-thor-base)|
|**FlanT5-large**| - |[🤗 nicolay-r/flan-t5-tsa-thor-large](https://huggingface.co/nicolay-r/flan-t5-tsa-thor-large) |
|**FlanT5-xl**   | [🤗 nicolay-r/flan-t5-tsa-prompt-xl](https://huggingface.co/nicolay-r/flan-t5-tsa-prompt-xl) | [🤗 nicolay-r/flan-t5-tsa-prompt-xl](https://huggingface.co/nicolay-r/flan-t5-tsa-thor-xl)|


# Three Hop Chain-of-Thought THoR  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework/blob/main/Reasoning_for_Sentiment_Analysis_Framework.ipynb)

```bash
python thor_finetune.py -r "thor" -d "rusentne2023" 
    -m "google/flan-t5-base" \
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
* `-m`, `--model_path`: Path to the model on hugging face.
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
