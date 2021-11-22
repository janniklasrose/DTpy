from dataclasses import dataclass


@dataclass
class Spin:
    """Nuclear Magnetic Spin."""
    gamma: float  # gyromagnetic ratio (rad/s/T)

# Define some common spins

# 1H (H+) Spin
# Value from <http://physics.nist.gov/cgi-bin/cuu/Value?gammap>
Proton = Spin(2.6752218744e8)

# Hyperpolarized Xenon 129
# Value from <https://en.wikipedia.org/wiki/Gyromagnetic_ratio#For_a_nucleus>
Xenon = Spin(-73.997e8)
