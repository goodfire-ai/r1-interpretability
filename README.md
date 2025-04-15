# Open-source SAEs for DeepSeek-R1

[[`Blog`]()] [[`Models`](https://huggingface.co/Goodfire/DeepSeek-R1-SAE-l37)]
[[`Dataset`](https://huggingface.co/Goodfire/r1-collect)] [[`Feature
Database`](https://huggingface.co/Goodfire/r1-collect)]

We're open-sourcing [two state-of-the-art
SAEs](https://huggingface.co/Goodfire/DeepSeek-R1-SAE-l37) trained on the 671B
parameter DeepSeek R1. These are the **first public interpreter models** trained
on a true reasoning model, and on **any model of this scale.** Because R1 is a
very large model and therefore difficult to run for most independent
researchers, we're also uploading SQL databases containing the max activating
examples for each feature.

We're excited to see how the wider research community
will use these tools to develop new techniques for understanding and aligning
powerful AI systems. As reasoning models continue to grow in capability and
adoption, tools like these will be essential for ensuring they remain reliable,
transparent, and aligned with human intentions. 

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

See `db_example.ipynb` for an example of how to interact with the database with
max activating examples for each feature. 