"""Bounded host submission facts; return is not device completion."""


class SubmissionJournal:
    def __init__(self, descriptors):
        self.descriptors = tuple(dict(item) for item in descriptors)
        self.states = ["NOT_ENTERED"] * len(self.descriptors)
        self.failed = False
        self.pending_device_operation = None

    def enter(self, index):
        if self.failed or self.states[index] != "NOT_ENTERED":
            raise RuntimeError("SUBMISSION_JOURNAL_REUSE")
        self.states[index] = "ENTERED"

    def returned(self, index):
        if self.failed or self.states[index] != "ENTERED":
            raise RuntimeError("SUBMISSION_JOURNAL_BOUNDARY")
        self.states[index] = "PUBLIC_RETURNED"

    def call(self, index, operation, *args, **kwargs):
        self.enter(index)
        result = operation(*args, **kwargs)
        self.returned(index)
        return result

    def device_call(self, name, operation, *args, **kwargs):
        """Track an ancillary GPU API boundary, not an extra logical child."""
        if self.failed or self.pending_device_operation is not None:
            raise RuntimeError("SUBMISSION_JOURNAL_REUSE")
        self.pending_device_operation = name
        result = operation(*args, **kwargs)
        self.pending_device_operation = None
        return result

    def failure(self):
        self.failed = True
        return tuple(
            dict(descriptor, submission=state)
            for descriptor, state in zip(self.descriptors, self.states)
        )

    def known_prefix(self):
        return tuple(
            descriptor
            for descriptor, state in zip(self.descriptors, self.states)
            if state == "PUBLIC_RETURNED"
        )

    def unknown(self):
        return self.pending_device_operation is not None or any(
            state == "ENTERED" for state in self.states
        )
