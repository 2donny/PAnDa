"""Prompt helpers kept in sync with the complete evaluator implementation."""


def build_truthfulqa_prompt(question):
    return question + "\nAnswer with one of the provided answer options only."


def build_strategyqa_prompt(question):
    return question + "\nAnswer with exactly one word: yes or no."


def build_gsm8k_prompt(question):
    return question + "\nAnswer with only the final number."


def build_gsm8k_sequence_prompt(question):
    return (
        question
        + "\nSolve this step by step in 2-4 short lines."
        + "\nEnd with exactly: Final answer: <number>"
    )


def build_halueval_prompt(task_name, knowledge, user_input, response_text, label_yes="A", label_no="B"):
    answer_instruction = (
        "Does the answer contain hallucinations?\n"
        f"Reply with exactly one label: {label_yes} = yes, {label_no} = no."
    )
    response_instruction = (
        "Does the response contain hallucinations?\n"
        f"Reply with exactly one label: {label_yes} = yes, {label_no} = no."
    )
    summary_instruction = (
        "Does the summary contain hallucinations?\n"
        f"Reply with exactly one label: {label_yes} = yes, {label_no} = no."
    )
    if task_name == "qa":
        return (
            f"Knowledge: {knowledge}\n"
            f"Question: {user_input}\n"
            f"Answer: {response_text}\n"
            f"{answer_instruction}"
        )
    if task_name == "dialogue":
        return (
            f"Knowledge: {knowledge}\n"
            f"Dialogue history: {user_input}\n"
            f"Response: {response_text}\n"
            f"{response_instruction}"
        )
    if task_name == "summarization":
        return (
            f"Document: {knowledge}\n"
            f"Summary: {response_text}\n"
            f"{summary_instruction}"
        )
    return (
        f"Context: {knowledge}\n"
        f"Prompt: {user_input}\n"
        f"Response: {response_text}\n"
        f"{response_instruction}"
    )


def build_pairwise_preference_prompt(prompt, candidate_a, candidate_b):
    return (
        "Compare two candidate answers to the same question.\n"
        "Choose the answer that is more likely to be correct.\n\n"
        f"Question:\n{prompt}\n\n"
        f"Candidate A:\n{candidate_a}\n\n"
        f"Candidate B:\n{candidate_b}\n\n"
        "Reply with exactly one label:\n"
        "A = Candidate A is more likely correct\n"
        "B = Candidate B is more likely correct\n"
    )
