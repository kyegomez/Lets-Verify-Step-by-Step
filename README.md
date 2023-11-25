# "Let’s Verify Step by Step"
Implementation of "Improving Mathematical Reasoning with Process Supervision" by OPENAI 

## Install
`pip3 install --upgrade process-supervision-torch`


## Usage:
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




