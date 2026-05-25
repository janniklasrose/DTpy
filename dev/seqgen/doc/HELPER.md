Generators
* https://github.com/imr-framework/pypulseq
* https://pulseq.github.io/

Others
* https://ismrm.github.io/mrhub/
* https://github.com/leoliuf/MRiLab
* https://github.com/nipy/dipy
* https://qmrlab.readthedocs.io/en/master/index.html#







# Sequence Generator

Inputs:
- $b$-value
- Maxiumum gradient strength $G_\text{max}$
- Slew rate, RF pulse duration
- Optimisation target: [Diffusion time $\Delta$ | Echo time TE | Gradient duration $\delta$]
- Sequence type & additional parameters
    - PGSE
    - M2-SE
    - STEAM
        - Target diffusion time $\Delta$

Output:
- Sequence $G(t)$
