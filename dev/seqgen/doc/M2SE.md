## DERIVATION for MCSE:

how????
`TM = 2tau + (del1 - 2tau)`    [1a]

from diagram
`TM = DELTA - del1 - del2`     [1b]

combine
`DELTA = 2del1 + del2`         [1]

condition on del2 = f(del1):
`del2*(DELTA - 2del1 - tau) = del1*(DELTA - tau)`    [2]

substitute [1] into [2], simplify, get polynomial in del2
`del2^2 + (tau - del1)*del2 + (tau*del1 - 2*del1^2) = 0`    [3]



should be
`del2 = (del1*(DELTA-tau)/(DELTA-2*del1+tau))`

___

manually calculate b-value
```
b(T) = int_0^T{ |k(t)|^2 dt }
k(t) = gamma*int_0^t{ G(tau) dtau }
```

for random pulse sequence
```
dt = ones(1, NT)*T/NT;
gG = rand(1, NT);
b = sum( norm(cumsum(gG .* dt)).^2 .* dt )
```
