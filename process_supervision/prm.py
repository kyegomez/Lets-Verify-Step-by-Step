from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead


class PRM:
    def __init__(
        self,
        model_name: str = "lvwerra/gpt2-imdb-pos-v2",
        ref_model_name: str = "lvwerra/gpt2-imdb",
        reward_model_name: str = "lvwerra/distilbert-imdb",
        device=None,
    ):
        """
        Initialize the PRM model with specified models and tokenizer.

        Args:
            model_name (str): Name of the main model.
            ref_model_name (str): Name of the reference model.
            reward_model_name (str): Name of the reward model.
            device (int or str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.ref_model_name = ref_model_name
        self.reward_model_name = reward_model_name
        self.device = device
        
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name
        ).to(device)

        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            ref_model_name
        ).to(device)

        self.reward_pipe = pipeline(
            "sentiment-analysis", model=reward_model_name, device=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_responses(
        self, queries: List[str], gen_len: int, gen_kwargs: Dict[str, Any]
    ) -> List[str]:
        """
        Generate responses for a batch of queries.

        Args:
            queries (list of str): List of query strings.
            gen_len (int): Length of the generated response.
            gen_kwargs (dict): Additional keyword arguments for generation.

        Returns:
            list of str: Generated responses.
        """
        responses = []
        for query in queries:
            input_ids = self.tokenizer.encode(query, return_tensors="pt").to(
                self.device
            )
            output_ids = self.model.generate(
                input_ids, max_new_tokens=gen_len, **gen_kwargs
            )
            response = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            responses.append(response)
        return responses

    def score_responses(
        self, responses: List[str], sent_kwargs: Dict[str, Any]
    ) -> List[float]:
        """
        Score a batch of responses using the reward pipeline.

        Args:
            responses (list of str): List of response strings.
            sent_kwargs (dict): Additional keyword arguments for sentiment analysis.

        Returns:
            list of float: Scores for each response.
        """
        scores = [
            output[0]["score"]
            for output in self.reward_pipe(responses, **sent_kwargs)
        ]
        return scores
