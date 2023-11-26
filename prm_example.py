import os

import torch
from dotenv import load_dotenv
from swarms.models import OpenAIChat

from process_supervision.generator import MathDataGenerator
from process_supervision.prm import PRM

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
