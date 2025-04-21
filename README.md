# Open-source SAEs for DeepSeek-R1

[[`Blog`](https://www.goodfire.ai/blog/under-the-hood-of-a-reasoning-model)]
[[`Models`](https://huggingface.co/Goodfire/DeepSeek-R1-SAE-l37)]
[[`Dataset`](https://huggingface.co/datasets/Goodfire/r1-collect)] 

We're open-sourcing [two state-of-the-art
SAEs](https://huggingface.co/Goodfire/DeepSeek-R1-SAE-l37) trained on the 671B
parameter DeepSeek R1. These are the **first public interpreter models** trained
on a true reasoning model, and on **any model of this scale.** Because R1 is a
very large model and therefore difficult to run for most independent
researchers, we're also uploading SQL databases containing hundreds of millions of tokens of activating
examples for each SAE.

We're excited to see how the wider research community will use these tools to
develop new techniques for understanding and aligning powerful AI systems. As
reasoning models continue to grow in capability and adoption, tools like these
will be essential for ensuring they remain reliable, transparent, and aligned
with human intentions. 

## Model Information

This release contains two SAEs, one for general reasoning and one for math, both
of which are [available on
HuggingFace](https://huggingface.co/Goodfire/DeepSeek-R1-SAE-l37). Load them
with the following snippet:

```python
from sae import load_math_sae
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id=f"Goodfire/DeepSeek-R1-SAE-l37",
    filename=f"math/DeepSeek-R1-SAE-l37.pt",
    repo_type="model"
)
device = "cpu"
math_sae = load_math_sae(file_path, device)
```

An example of loading and inference for both SAEs is available in `sae_example.ipynb`.

The general reasoning SAE was trained on R1's activations on our [custom
reasoning dataset](https://huggingface.co/datasets/Goodfire/r1-collect), and the second
used [OpenR1-Math](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), a
large dataset for mathematical reasoning. These datasets allow us to discover
the features that R1 uses to answer challenging problems that exercise its
reasoning chops.

## Feature Database

To help researchers use these SAEs, we're publishing autointerped feature labels and feature activations on hundreds of millions of tokens.
The feature labels are available as a SQL database or a CSV, while the feature activations are available as a SQL database.
To download them, use the following s3 links:

### Math SAE

Autointerp labels: [CSV](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/math/autointerp.csv) or [SQL](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/math/autointerp.db)

Feature activations & their corresponding tokens
  | Sample                                                                          | Tokens | Size  |
  | :------------------------------------------------------------------------------ | :----- | :---- |
  | [Full](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/math/math.ddb)     | 521M   | 440GB |
  | [10%](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/math/math-10.ddb)   | 52.1M  | 47GB  |
  | [1%](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/math/math-1.ddb)    | 5.21M  | 7GB   |
  | [0.1%](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/math/math-0-1.ddb) | 521K   | 3GB   |

### Logic SAE

  Autointerp labels: [CSV](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/logic/autointerp.csv) or [SQL](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/logic/autointerp.db)

  Feature activations & their corresponding tokens
  | Sample                                                                            | Tokens | Size  |
  | :-------------------------------------------------------------------------------- | :----- | :---- |
  | [Full](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/logic/logic.ddb)     | 219M   | 123GB |
  | [10%](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/logic/logic-10.ddb)   | 21.9M  | 13GB  |
  | [1%](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/logic/logic-1.ddb)    | 2.19M  | 2GB   |
  | [0.1%](https://goodfire-r1-features.s3.us-east-1.amazonaws.com/logic/logic-0-1.ddb) | 219K   | 1GB   |

Logic SAE

See `db_example.ipynb` for examples of interacting with the databases.

## R1-Collect

We collected a large dataset of R1-generated tokens on various open-source
reasoning and logic datasets. These were collected from:

- [DeepMind Code
  Contests](https://huggingface.co/datasets/deepmind/code_contests)
- [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa)
- [AGIEval](https://huggingface.co/datasets/lighteval/agi_eval_en)
- [gaokao-bench](https://huggingface.co/datasets/RUCAIBox/gaokao-bench)
