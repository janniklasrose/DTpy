from .sequence import Sequence


class M2SE():
    """Motion Compensated Spin Echo."""

    def __init__(self, *args, **kwargs) -> None:
        self.sequence = Sequence()
        raise NotImplementedError("M2SE sequence generation is not implemented yet")
