# Balls and Bins: Variable Neighborhood Search and an Exact MIP Model

Course project for *Combinatorial Optimization* (UFRGS). It contains two solvers for the
Balls and Bins problem:

- a **Variable Neighborhood Search (VNS)** metaheuristic in Python, and
- an **exact mixed-integer programming (MIP) model** in Julia, built with JuMP and solved with HiGHS.

## The problem

You have $B$ balls and $n$ bins, and bin $i$ must hold between $l_i$ and $u_i$ balls. A bin holding
$b$ balls is worth $1 + 2 + \dots + b = b(b+1)/2$; equivalently, the $k$-th ball in a bin is worth $k$.
The goal is to choose how many balls $b_i$ go into each bin so that the total profit is maximized:

$$
\max \sum_{i=1}^{n} \frac{b_i (b_i + 1)}{2}
\quad \text{s.t.} \quad
\sum_{i=1}^{n} b_i \le B, \qquad
l_i \le b_i \le u_i, \qquad
b_i \in \mathbb{Z}.
$$

The objective is convex, so this is maximization of a convex function. Concentrating balls in a few
bins pays off, but the bounds limit how far that can go.

### Instance format

Instance files hold whitespace-separated integers: the number of bins $n$, the number of balls $B$,
then one `lower upper` pair per bin. For example, [`instances/example.txt`](instances/example.txt)
starts like this:

```
10
67
2 12
5 10
...
```

## Exact model (MIP)

The profit is linearized with one binary variable per ball slot above the lower bound:
$x_{ij} = 1$ if and only if bin $i$ holds at least $l_i + j$ balls, for $j = 1, \dots, u_i - l_i$.
The $(l_i + j)$-th ball in bin $i$ adds $l_i + j$ to the profit, so the model is

$$
\begin{aligned}
\max\quad & \sum_{i} \frac{l_i (l_i + 1)}{2} + \sum_{i} \sum_{j=1}^{u_i - l_i} (l_i + j)\, x_{ij} \\
\text{s.t.}\quad & x_{ij} \le x_{i,j-1} && \forall i,\ j = 2, \dots, u_i - l_i \\
& \sum_{i} \sum_{j} x_{ij} \le B - \sum_{i} l_i \\
& x_{ij} \in \{0, 1\}.
\end{aligned}
$$

The ordering constraints $x_{ij} \le x_{i,j-1}$ are required. The marginal profit $l_i + j$ grows
with $j$, so without them the solver would pick the most valuable slots instead of a prefix of slots.

## Variable Neighborhood Search

- **Representation.** A solution stores how many balls each bin holds *above* its lower bound, so
  the lower bounds always hold. No move breaks an upper bound or the ball budget, so every
  solution the search visits is feasible.
- **Initial solution.** A greedy pass fills the bins to their upper bound, largest upper bound first,
  until the balls run out.
- **Neighborhoods.** A move takes one ball from a bin above its lower bound and puts it in another
  bin below its upper bound. Moving a ball from bin $i$ to bin $j$ changes the profit by
  $(b_j + 1) - b_i$, which is evaluated in $O(1)$. The neighborhood $N_k$ contains the solutions
  reachable with $k$ such moves, and *shaking* picks a random point of $N_k$.
- **Local search.** The local search uses sampled best improvement. Each iteration samples a set of
  distinct one-ball moves and applies the best one if it improves the profit. It stops at the first
  sample with no improving move, or after a fixed number of iterations.
- **Acceptance.** A candidate $x'$ replaces the current solution $x$ when
  $g(x') + \varepsilon > g(x)$, with $\varepsilon \sim \mathcal{N}(0, (\sigma\, g(x))^2)$. Here
  $g$ is the profit above the lower-bound baseline and $\sigma$ is the noise level. The noise
  sometimes accepts a slightly worse solution, which helps the search leave local optima. The best
  solution found so far is kept separately.
- **Adaptive neighborhood size.** A sweep tries $k = 1, \dots, k_{\max}$, and every acceptance
  restarts it at $k = 1$. Each acceptance also shrinks $k_{\max}$ by one, but never below its
  initial value. After a set number of consecutive sweeps without an acceptance, $k_{\max}$ grows
  by one, up to a limit.
- **Stopping.** The search stops after a maximum number of sweeps or a time limit.

## Repository layout

```
.
├── instances/
│   └── example.txt            small example instance
├── mip/
│   ├── Project.toml           Julia environment (JuMP, HiGHS)
│   └── balls_and_bins.jl      exact MIP model
├── vns/
│   └── vns_balls_and_bins.py  VNS metaheuristic
└── requirements.txt           Python dependencies
```

## Requirements

- **VNS:** Python 3.9+ and NumPy (`pip install -r requirements.txt`).
- **MIP:** Julia 1.6+. Install JuMP and HiGHS once with:

  ```bash
  julia --project=mip -e 'using Pkg; Pkg.instantiate()'
  ```

## Usage

### VNS

```bash
python vns/vns_balls_and_bins.py INSTANCE [options]
```

| Option | Short | Description | Default |
|---|---|---|---|
| `--seed` | `-s` | Random seed | `42` |
| `--iterations` | `-i` | Maximum number of sweeps | `1000` |
| `--timeout` | `-t` | Time limit in seconds | `3600` |
| `--max-neighborhood` | `-n` | Initial $k_{\max}$ (moves per shake) | `3` |
| `--neighborhood-limit` | `-nl` | Largest value $k_{\max}$ can grow to | `7` |
| `--max-stagnation` | `-g` | Sweeps without acceptance before $k_{\max}$ grows | `5` |
| `--noise-level` | `-x` | Acceptance noise $\sigma$, relative to the current gain | `0.05` |
| `--local-samples` | `-ls` | Moves sampled per local search iteration | `35` |
| `--local-iterations` | `-li` | Maximum local search iterations | `25` |
| `--output` | `-o` | Write the run as one JSON line to this file | none |
| `--append` | `-a` | Append to `--output` instead of overwriting it | off |
| `--quiet` | `-q` | Do not log improvements during the search | off |

Example:

```console
$ python vns/vns_balls_and_bins.py instances/example.txt -i 200
[    0.01s] greedy profit: 385
[    0.01s] new best profit: 401
[    0.02s] new best profit: 409
Instance:        instances/example.txt (10 bins, 67 balls)
Configuration:   max_neighborhood=3, neighborhood_limit=7, max_stagnation=5, noise_level=0.05, ...
Initial profit:  385 (greedy)
Best profit:     409
Balls per bin:   [12, 5, 16, 1, 11, 3, 1, 5, 0, 13]
Sweeps:          200
Time elapsed:    0.76 s
```

Progress lines go to stderr and the summary goes to stdout. Use `-o` with `-a` to collect several
runs in a [JSON Lines](https://jsonlines.org) file. Each record holds the configuration, the best
and initial solutions, the elapsed time and the improvement history:

```bash
for seed in 1 2 3 4 5; do
    python vns/vns_balls_and_bins.py instances/example.txt -t 60 -s "$seed" -q -o runs.jsonl -a
done
```

With a fixed seed, a run limited by `--iterations` is deterministic. A run that stops on
`--timeout` depends on the speed of the machine.

### MIP

```bash
julia --project=mip mip/balls_and_bins.jl INSTANCE [SEED=42] [TIME_LIMIT=3600]
```

After the HiGHS log, the script prints the termination status, the best profit and bound, the
optimality gap and the balls per bin:

```console
$ julia --project=mip mip/balls_and_bins.jl instances/example.txt
...
Instance:       instances/example.txt (10 bins, 67 balls)
Status:         OPTIMAL
Best profit:    409
Best bound:     409
Gap:            0.0%
Balls per bin:  [12, 5, 16, 1, 11, 3, 1, 5, 0, 13]
Solve time:     0.02 s
```

The model has one binary variable per ball slot, $\sum_i (u_i - l_i)$ in total. On large instances,
the time limit can expire before HiGHS proves optimality or even finds a feasible solution. The VNS
is meant for those instances.
