import torch 
from process_supervision.main import GPT4

# Usage with random inputs
text = torch.randint(0, 20000, (1, 1024))

# Initiliaze the model
model = GPT4()
output = model(text)
print(output)
