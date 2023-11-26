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
from process_supervision.prm import PRM
from swarms.models import OpenAIChat
from process_supervision.generator import MathDataGenerator
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# LLM initialization
llm = OpenAIChat(openai_api_key=api_key)

# Math data generator initialization
math_datagenerator = MathDataGenerator(llm, num_iters=10)

# Device initialization
device = 0 if torch.cuda.is_available() else "cpu"

# Model initialization
prm_model = PRM(
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
queries = [math_datagenerator.generate_samples(query) for query in queries]

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
- [ ] We need help integrating the math sample generator, first create the class and prompts and pass them into gpt4
- [ ] Then conduct best of N sampling with the reward model and reward each step
- [ ] Train or finetune now model with dataset
- [ ] Have a better idea? LMK


# License
MIT

