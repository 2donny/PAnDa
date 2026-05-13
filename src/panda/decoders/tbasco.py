"""Public wrappers for the TBASCo decoder."""


def choose_tbasco_branch(evaluator, prompt, low_prediction, high_prediction):
    from ..evaluation import choose_tbasco_branch as impl

    return impl(evaluator, prompt, low_prediction, high_prediction)


def generate_with_tbasco_decoder(evaluator, prompt, max_new_tokens=96, stop_on_eos=True):
    from ..evaluation import generate_with_tbasco_decoder as impl

    return impl(
        evaluator,
        prompt,
        max_new_tokens=max_new_tokens,
        stop_on_eos=stop_on_eos,
    )
