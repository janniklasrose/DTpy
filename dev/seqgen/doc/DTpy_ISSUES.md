# DTpy

## Pulse sequences

- Python to generate sequences
- Write to (custom, but intuitive) `.seq` file
- Python for post-processing again (DTpy)

## Sequence definition vs time steps

- Pulse sequence is $`G(t)`$, includes $`t=0`$ and $`t=T`$
- Simulation needs $`G(n)`$ and $`\delta t(n)`$
  - initial $`n=0`$ state, followed by NT steps
- what about sequences where $`G`$ does not fit nicely on uniform $`\delta t`$?
