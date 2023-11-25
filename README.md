# "Let’s Verify Step by Step"
Implementation of "Improving Mathematical Reasoning with Process Supervision" by OPENAI 

## Install
`pip3 install --upgrade process-supervision-torch`


## Usage:

### GPT4 without tokenizer
```python
import torch 
from process_supervision.main import GPT4

# Usage with random inputs
text = torch.randint(0, 20000, (1, 1024))

# Initiliaze the model
model = GPT4()
output = model(text)
print(output)
```


### `PRM`
```python
import torch
from process_supervision.main import PRM

device = 0 if torch.cuda.is_available() else "cpu"

# Model initialization
prm_model = PRMModel(
   model_name="lvwerra/gpt2-imdb-pos-v2",
   ref_model_name="lvwerra/gpt2-imdb",
   reward_model_name="lvwerra/distilbert-imdb",
   device=device,
)

# Generation arguments
gen_kwargs = {
   "min_length": -1,
   "top_k": 0.0,
   "top_p": 1.0,
   "do_sample": True,
   "pad_token_id": prm_model.tokenizer.eos_token_id,
}
sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

# Sample queries
queries = ["Sample query 1", "Sample query 2"]

# Generate responses
responses = prm_model.generate_responses(
    queries, gen_len=10, gen_kwargs=gen_kwargs
)

# Score responses
scores = prm_model.score_responses(responses, sent_kwargs)

# Display results
for query, response, score in zip(queries, responses, scores):
    print(f"Query: {query}\nResponse: {response}\nScore: {score}\n")

```


### GPT4 + PRM
```



# Method


# Citation
```bibtex
@misc{lightman2023lets,
   title={Let's Verify Step by Step}, 
   author={Hunter Lightman and Vineet Kosaraju and Yura Burda and Harri Edwards and Bowen Baker and Teddy Lee and Jan Leike and John Schulman and Ilya Sutskever and Karl Cobbe},
   year={2023},
   eprint={2305.20050},
   archivePrefix={arXiv},
   primaryClass={cs.LG}
}

```

# Todo
- [ ] Creae the PRM reward model




# License
MIT




