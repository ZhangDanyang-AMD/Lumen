"""Canonical physical children from the public prepared config view only."""


def dense_children(view, counts, pass_id, *, replay=False, experts=None):
    """Map all calls, or only proven returned experts, in physical order."""
    children = []
    for expert, count in enumerate(counts):
        if experts is not None and expert not in experts:
            continue
        operations = []
        if count:
            if pass_id == 0 and not replay:
                operations.append(("X_dual", view.primitive(2, variant=2)))
            if pass_id == 1:
                operations.append(("dY_dual", view.primitive(2, variant=3)))
            operations.append(("gemm", view.gemm(pass_id)))
        elif pass_id == 2 and view.w_output_mode == 0:
            operations.append(
                ("zero", view.primitive(5, variant=4, output_dtype=view.w_output_dtype))
            )
        for operation, config in operations:
            children.append(
                {
                    "op": operation,
                    "expert": expert,
                    "config_id": list(config.config),
                    "ordinal": len(children),
                }
            )
    return children
