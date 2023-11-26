from unittest.mock import Mock

import pytest
from transformers import AutoModelForCausalLMWithValueHead, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from process_supervision.prm import PRM


@pytest.fixture
def prm_model():
    return PRM(
        model_name="lvwerra/gpt2-imdb-pos-v2",
        ref_model_name="lvwerra/gpt2-imdb",
        reward_model_name="lvwerra/distilbert-imdb",
        device="cpu",
    )

def test_prm_model_init(prm_model):
    assert prm_model.model_name == "lvwerra/gpt2-imdb-pos-v2"
    assert prm_model.ref_model_name == "lvwerra/gpt2-imdb"
    assert prm_model.reward_model_name == "lvwerra/distilbert-imdb"
    assert prm_model.device == "cpu"
    assert isinstance(prm_model.model, AutoModelForCausalLMWithValueHead)
    assert isinstance(prm_model.ref_model, AutoModelForCausalLMWithValueHead)
    assert isinstance(prm_model.tokenizer, AutoTokenizer)

def test_generate_responses(prm_model):
    queries = ["How are you?", "What is the weather today?"]
    gen_len = 10
    gen_kwargs = {"do_sample": True}
    responses = prm_model.generate_responses(queries, gen_len, gen_kwargs)
    assert isinstance(responses, list)
    assert len(responses) == len(queries)
    for response in responses:
        assert isinstance(response, str)

def test_score_responses(prm_model):
    responses = ["I'm good.", "The weather is sunny."]
    sent_kwargs = {"truncation": True}
    scores = prm_model.score_responses(responses, sent_kwargs)
    assert isinstance(scores, list)
    assert len(scores) == len(responses)
    for score in scores:
        assert isinstance(score, float)

@pytest.mark.parametrize("queries, gen_len, gen_kwargs", [
    (["Hello"], 5, {"do_sample": False}),
    (["How are you?", "What is the weather today?"], 15, {"do_sample": True}),
])
def test_generate_responses_parametrized(prm_model, queries, gen_len, gen_kwargs):
    responses = prm_model.generate_responses(queries, gen_len, gen_kwargs)
    assert isinstance(responses, list)
    assert len(responses) == len(queries)
    for response in responses:
        assert isinstance(response, str)

@pytest.mark.parametrize("responses, sent_kwargs", [
    (["I'm good.", "The weather is sunny."], {"truncation": True}),
    (["Great!", "It's raining."], {"truncation": False}),
])
def test_score_responses_parametrized(prm_model, responses, sent_kwargs):
    scores = prm_model.score_responses(responses, sent_kwargs)
    assert isinstance(scores, list)
    assert len(scores) == len(responses)
    for score in scores:
        assert isinstance(score, float)

def test_generate_responses_with_mocked_model(prm_model, monkeypatch):
    mock_generate = Mock(return_value=[[1, 2, 3]])
    monkeypatch.setattr(prm_model.model, "generate", mock_generate)
    queries = ["How are you?"]
    gen_len = 5
    gen_kwargs = {"do_sample": True}
    responses = prm_model.generate_responses(queries, gen_len, gen_kwargs)
    assert responses == ["[CLS] [SEP] [PAD] [PAD] [PAD]"]

def test_score_responses_with_mocked_pipe(prm_model, monkeypatch):
    mock_pipe = Mock(return_value=[{"score": 0.8}])
    monkeypatch.setattr(prm_model.reward_pipe, "__call__", mock_pipe)
    responses = ["I'm good."]
    sent_kwargs = {"truncation": True}
    scores = prm_model.score_responses(responses, sent_kwargs)
    assert scores == [0.8]

def test_generate_responses_exception(prm_model):
    with pytest.raises(Exception):
        prm_model.generate_responses("Hello", 5, {"do_sample": False})

def test_score_responses_exception(prm_model):
    with pytest.raises(Exception):
        prm_model.score_responses("Great!", {"truncation": False})
