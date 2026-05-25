## Sequence Diagrams

### PGSE
```
   0             _____          @TE/2          _____             TE
___|___         /     \      _____|_____      /     \     ________|________
alpha90   x1             x2    alpha180   x3           x4     alphaEcho
```

0) Constants: alpha90, alpha180, alphaEcho ??
1) Input: b, G, e'
   Compute: e = G/e'
2) Guess: D
   Compute: d=f(b,D,e,G), x1,x2,x3,x4

-> Minimise TE








### M2SE
Same as PGSE, just gradients are more complicated








### STEAM

```
           ___                                     ___
__|__     /   \       __|__      ~     __|__      /   \      ______|______
```

Similar to PGSE, but we have 3x 90deg pulses
x1,x2,x3,x4,x5

