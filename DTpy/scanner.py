from dataclasses import dataclass


@dataclass
class Scanner:
    #TODO: are these per axis?
    gradientStrength: float  # T/m
    slewRate: float  # T/m/s

DefaultScanner = Scanner(40e-3, 60) # 40 mTm, 60 T/m/s
