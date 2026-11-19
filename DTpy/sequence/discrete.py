from .sequence import Sequence


class DiscreteSequence(Sequence):
    """Discrete values."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Discrete sequence generation is not implemented yet")


def discretise_sequence(*args, **kwargs):
    raise NotImplementedError("Sequence discretisation is not implemented yet")
