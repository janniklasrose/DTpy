from dataclasses import dataclass


@dataclass
class Scanner:
    gradientStrength: float  # T/m
    slewRate: float  # T/m/s
