from .sequence import Sequence


class PGSE():
    """Pulse Gradient Spin Echo."""

    def __init__(self, *args, **kwargs) -> None:
        self.sequence = Sequence()
        raise NotImplementedError("PGSE sequence generation is not implemented yet")
