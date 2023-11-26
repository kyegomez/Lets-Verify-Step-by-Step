import os

from dotenv import load_dotenv
from swarms.models import OpenAIChat

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# LLM initialization
llm = OpenAIChat(api_key=api_key)


class MathDataGenerator:
    """
    Math data generator for the LLM.

    Args:
        llm (OpenAIChat): LLM model.
        num_iters (int): Number of iterations to run the LLM.

    Returns:
        list of dict: Generated samples.

    Examples:
        >>> llm = OpenAIChat(api_key=api_key)
        >>> mdg = MathDataGenerator(llm, num_iters=10)
        >>> mdg.generate_samples("1 + 1 = 2")
        [{'query': '1 + 1 = 2', 'response': '1 + 1 = 2', 'score': 0.0, 'reward': 0.0}]

    """

    def __init__(self, llm, num_iters):
        self.llm = llm
        self.num_iters = num_iters

    def generate_samples(self, task: str):
        """Generate samples for a given task.

        Args:
            task (str): _description_

        Returns:
            _type_: _description_
        """
        memory = []
        for _ in range(self.num_iters):
            results = self.llm(task)
            memory.append(results)
        return memory
