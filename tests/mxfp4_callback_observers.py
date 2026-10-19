"""Callbacks shared by real Core fixtures and delayed-binding host regressions."""


def capture_probability(destination, key):
    def hook(_module, args):
        probability = args[2]
        destination[key] = probability
        probability.retain_grad()

    return hook


def capture_router_map(destination):
    def hook(_module, _args, result):
        destination["map"] = result[1].detach().clone()

    return hook


def gradient_emitter(handoff, lease):
    def backward(_ctx, _gradient):
        return handoff.arm(lease)

    return backward
