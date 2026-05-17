from dataclasses import dataclass


@dataclass
class Scanner:
    #TODO: are these per axis?
    gradientStrength: float  # T/m
    slewRate: float  # T/m/s
