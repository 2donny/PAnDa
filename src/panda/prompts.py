"""Prompt helpers kept in sync with the public TruthfulQA evaluator."""


def build_truthfulqa_prompt(question):
    return question + "\nAnswer with one of the provided answer options only."
