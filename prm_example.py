import torch
from process_supervision.prm import PRM
from swarms.models import OpenAIChat
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# LLM initialization
llm = OpenAIChat(api_key=api_key)

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

# Generate responses
responses = prm_model.generate_responses(
    queries, gen_len=10, gen_kwargs=gen_kwargs
)

# Score responses
scores = prm_model.score_responses(responses, sent_kwargs)

# Display results
for query, response, score in zip(queries, responses, scores):
    print(f"Query: {query}\nResponse: {response}\nScore: {score}\n")
