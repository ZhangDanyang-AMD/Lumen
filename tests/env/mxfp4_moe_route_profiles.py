"""Deterministic distinct-top6 target schedules; pure host planning, real router execution."""


def targets(profile, token, tokens, sender, ep):
    global_token = sender * tokens + token
    if profile == "balanced":
        start = (global_token * 6) % 256
    elif profile == "half_empty":
        start = (global_token * 6) % 128
    elif profile == "many_empty":
        start = (global_token * 6) % 24
    elif profile == "single_hot":
        start = 0
    elif profile == "zipf":
        # Deterministic dyadic heavy tail over contiguous six-expert groups.
        number = global_token + 1
        trailing = 0
        while number % 2 == 0:
            trailing += 1
            number //= 2
        start = min(trailing, 41) * 6
    else:
        raise ValueError(profile)
    capacity = (
        128 if profile == "half_empty" else 24 if profile == "many_empty" else 256
    )
    return tuple((start + index) % capacity for index in range(6))


EXACT_M = (0, 1, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 257)


def exact_plan(ep, cell, profile=None):
    """Receiver0 exact ragged counts; only remote experts fill remaining top6."""
    if ep not in (4, 8) or cell not in (0, 1):
        raise ValueError("signed EP/cell required")
    local = 256 // ep
    vector = list(EXACT_M) + [31, 1, 33] + [0] * (local - 16)
    if profile is not None:
        profiles = {
            "balanced": [4] * local,
            "half_empty": [4] * (local // 2) + [0] * (local // 2),
            "many_empty": [7, 5, 3] + [0] * (local - 3),
            "single_hot": [33] + [0] * (local - 1),
            "zipf": [max(1, 64 // (index + 1)) for index in range(local)],
            "all_empty": [0] * local,
        }
        vector = profiles[profile]
    if cell:
        # Rotate as a unit, retaining the adjacent 31/1/33 sentinel.
        shift = 16 if profile is None else 1
        vector = vector[shift:] + vector[:shift]
    remote = tuple(range(local, local + 6))
    routed = []
    for expert, count in enumerate(vector):
        routed.extend((expert, *remote[:5]) for _ in range(count))
    senders = [routed[sender::ep] for sender in range(ep)]
    rows = max(map(len, senders))
    for sender in senders:
        sender.extend([remote] * (rows - len(sender)))
    counts = [0] * 256
    for sender in senders:
        for token in sender:
            for expert in token:
                counts[expert] += 1
    assert tuple(counts[:local]) == tuple(vector)
    return tuple(tuple(sender) for sender in senders), tuple(counts)


def expected_counts(profile, tokens, ep, receiver):
    local = 256 // ep
    counts = [0] * local
    for sender in range(ep):
        for token in range(tokens):
            for expert in targets(profile, token, tokens, sender, ep):
                if receiver * local <= expert < (receiver + 1) * local:
                    counts[expert - receiver * local] += 1
    return tuple(counts)
