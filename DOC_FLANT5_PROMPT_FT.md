# Flan-T5 fine-tuning findings on RuSentNE-2023

We use the following prompt message:

```
What's the attitude of the sentence '{context}' to the target '{target}'? Select one from: positive, negative, neutral.
```

**Setup**: `Flan-T5-base`, output up to 300 tokens, 12 epochs, 16-batch size. 
NVidia-V100, ~1.5 min/epoch

**Implementation:** `engine_prompt.py`

**Result:** F1_PN = 57.01

**Setup**: `Flan-T5-large` output up to 300 tokens, 10 epochs. 9'th is the best. 16-batch size. NVidia-A100, ~1.5 min/epoch

**Implementation:** `engine_prompt.py`

**Result:** F1_PN = 60.796
```tsv
     F1_PN  F1_PN0  default   mode
0   62.009  70.023   70.023  valid
1   64.580  72.050   72.050  valid
2   65.444  73.350   73.350  valid
3   64.866  72.894   72.894  valid
4   65.378  73.474   73.474  valid
5   65.145  73.261   73.261  valid
6   65.321  73.363   73.363  valid
7   64.909  72.898   72.898  valid
8   65.175  73.009   73.009  valid
9   65.831  73.706   73.706  valid
10  60.796  69.792   69.792   test
11  60.796  69.792   69.792   test
```

**Setup**: `Flan-T5-xl` trained for 4 epochs. Model has not been even overfitted after 4 epochs! This is the state-of-the-art of the prompt tuning 
Trained with the new version of the project: 

**Command for reproduction:**
```bash
python main.py -r prompt -d rusentne2023 -bs 4 -es 4 -bf16 -p "What's the attitude of the sentence '{context}', to the target '{target}'?"
```

**Checkpoint**: [🤗 nicolay-r/flan-t5-tsa-prompt-large](https://huggingface.co/nicolay-r/flan-t5-tsa-prompt-xl)

**Result:** F1_PN = **68.197** 
```tsv
   F1_PN  F1_PN0  default   mode
0  63.254  71.802   71.802  valid
1  67.996  75.173   75.173  valid
2  68.182  75.240   75.240  valid
3  68.624  75.678   75.678  valid
4  68.197  75.290   75.290   test
5  68.197  75.290   75.290   test
```
