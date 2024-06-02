## Flan-T5 THOR fine-tuning findings on RuSentNE-2023 

The provided logging information here is for the prompts of version 1 in english.

**Setup:** `Flan-T5-base`, output up to 300 tokens, 5 epochs, 16-batch size.

**GPU:** `NVidia-A100`, ~4 min/epoch, temperature 1.0, float 32

**Checkpoint:** [🤗 nicolay-r/flan-t5-tsa-thor-base](https://huggingface.co/nicolay-r/flan-t5-tsa-thor-base)

**Result:** F1_PN = 60.024
```tsv
    F1_PN  F1_PN0  default   mode
0  45.523  59.375   59.375  valid
1  62.345  70.260   70.260  valid
2  62.722  70.704   70.704  valid
3  62.721  70.671   70.671  valid
4  62.357  70.247   70.247  valid
5  60.024  68.171   68.171   test
6  60.024  68.171   68.171   test
```
> It is possible to see, that since epoch #2 model is already fine-tuned.

**Setup:** `Flan-T5-large`, output up to 300 tokens, 5 epochs, 12-batch size. NVidia-A100, ~12 min/epoch, temperature 1.0, float 32

**Checkpoint:** [🤗 nicolay-r/flan-t5-tsa-thor-large](https://huggingface.co/nicolay-r/flan-t5-tsa-thor-large)

**Result:** F1_PN = 62.715

```tsv
    F1_PN  F1_PN0  default   mode
0  60.270  69.261   69.261  valid
1  66.226  73.596   73.596  valid
2  65.704  73.675   73.675  valid
3  66.729  74.186   74.186  valid
4  67.314  74.669   74.669  valid
5  62.715  71.001   71.001   test
6  62.715  71.001   71.001   test
```

**Setup:** `Flan-T5-xl` up to 64 tokens, 4-batch size, NVidia-A100, **bfloat16** ~ 30 min/epoch, NVidia A100 (40GB)
 
**Result:** Confirmed as mentioned in THoR paper. Reasoning capabilities are even higher than with `large` model.
The best result from the epoch #2.

**Checkpoint:** [🤗 nicolay-r/flan-t5-tsa-prompt-xl](https://huggingface.co/nicolay-r/flan-t5-tsa-thor-xl)

**Result:** F1_PN = 65.090
```tsv
  F1_PN  F1_PN0  default   mode
0  66.678  73.838   73.838  valid
1  68.019  74.816   74.816  valid
2  67.870  74.688   74.688  valid
3  65.090  72.449   72.449   test
4  65.090  72.449   72.449   test
```