from .sequence import Sequence


class Constant():
    """Simple constant-gradient with fixed time step."""

    def __init__(self, NT: int, dt: float, gradient: float = 0) -> None:
        if NT < 0:
            raise ValueError("NT must be non-negative")

        self.dt = [dt] * NT
        self.grad = [gradient] * NT

        timepoints = [index * dt for index in range(NT)]
        gradients = [gradient] * NT
        self.sequence = Sequence(timepoints=timepoints, gradients=gradients)
