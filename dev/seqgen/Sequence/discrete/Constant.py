from Sequence import Sequence

class Constant(Sequence):
    """Simple sequence"""

    def set_sequence(self, NT, dt):
        self.dt = [dt]*NT
        self.grad = [(0,0,0)]*NT
