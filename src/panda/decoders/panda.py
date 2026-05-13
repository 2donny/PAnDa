"""Public wrappers for the PAnDa decoder."""


def run_panda_block(evaluator, generated, window_size):
    return evaluator.run_panda_block(generated, window_size)


def generate_with_panda_decoder(evaluator, prompt, max_new_tokens=96, stop_on_eos=True):
    return evaluator.generate_with_panda_decoder(
        prompt,
        max_new_tokens=max_new_tokens,
        stop_on_eos=stop_on_eos,
    )
