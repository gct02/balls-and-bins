"""Variable Neighborhood Search (VNS) for the Balls and Bins problem.

Problem
-------
Distribute at most ``B`` balls among ``n`` bins. Bin ``i`` must hold between
``l_i`` and ``u_i`` balls, and a bin holding ``b`` balls is worth
``1 + 2 + ... + b = b (b + 1) / 2``. The goal is to maximize the total profit.

Representation
--------------
The lower bounds are always met, so a solution only stores how many balls each
bin holds *above* its lower bound (``0 <= extra_i <= u_i - l_i``). The search
moves single balls between bins; moving one ball from bin ``i`` to bin ``j``
changes the profit by ``(l_j + extra_j + 1) - (l_i + extra_i)``, which is
evaluated in O(1).

Usage
-----
    python vns_balls_and_bins.py INSTANCE [options]      (see --help)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Number of random draws made when sampling distinct moves for local search.
MAX_SAMPLING_TRIES = 1000


def triangular(n):
    """Profit of a bin holding ``n`` balls: 1 + 2 + ... + n."""
    return n * (n + 1) // 2


# ---------------------------------------------------------------------------
# Instance and solution
# ---------------------------------------------------------------------------


@dataclass
class Instance:
    """A Balls and Bins instance: a ball budget and per-bin [lower, upper] bounds."""

    n_balls: int
    lower: np.ndarray
    upper: np.ndarray
    capacity: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=np.int64)
        self.upper = np.asarray(self.upper, dtype=np.int64)
        if self.lower.shape != self.upper.shape or self.lower.ndim != 1:
            raise ValueError("lower and upper must be 1-D arrays of the same length")
        if np.any(self.lower < 0) or np.any(self.upper < self.lower):
            raise ValueError("every bin needs 0 <= lower <= upper")
        if self.free_balls < 0:
            raise ValueError(
                f"infeasible instance: the lower bounds need {int(self.lower.sum())} "
                f"balls but only {self.n_balls} are available"
            )
        self.capacity = self.upper - self.lower

    @classmethod
    def from_file(cls, path: str | Path) -> Instance:
        """Read an instance: ``n_bins``, ``n_balls``, then one ``lower upper`` line per bin."""
        tokens = Path(path).read_text().split()
        n_bins, n_balls = int(tokens[0]), int(tokens[1])
        bounds = np.array(tokens[2:], dtype=np.int64)
        if bounds.size != 2 * n_bins:
            raise ValueError(f"{path}: expected {n_bins} bounds pairs, found {bounds.size / 2:g}")
        bounds = bounds.reshape(n_bins, 2)
        return cls(n_balls, bounds[:, 0], bounds[:, 1])

    @property
    def n_bins(self) -> int:
        return self.lower.size

    @property
    def free_balls(self) -> int:
        """Balls left to distribute once every lower bound is met."""
        return self.n_balls - int(self.lower.sum())

    @property
    def base_profit(self) -> int:
        """Profit of placing exactly the lower bound in every bin."""
        return int(triangular(self.lower).sum())


class Solution:
    """Balls placed above each bin's lower bound, with the profit they add.

    ``gain`` is the profit above ``instance.base_profit`` and is updated
    incrementally by :meth:`apply_move`.
    """

    def __init__(self, instance: Instance, extra: np.ndarray, gain: int | None = None):
        self.instance = instance
        self.extra = extra
        self.gain = self.evaluate() if gain is None else gain

    def evaluate(self) -> int:
        """Recompute the gain from scratch."""
        lower = self.instance.lower
        return int((triangular(lower + self.extra) - triangular(lower)).sum())

    @property
    def profit(self) -> int:
        return self.instance.base_profit + self.gain

    def balls_per_bin(self) -> list[int]:
        return (self.instance.lower + self.extra).tolist()

    def copy(self) -> Solution:
        return Solution(self.instance, self.extra.copy(), self.gain)

    def donors(self) -> np.ndarray:
        """Bins that can give a ball (above their lower bound)."""
        return np.flatnonzero(self.extra > 0)

    def receivers(self) -> np.ndarray:
        """Bins that can take a ball (below their upper bound)."""
        return np.flatnonzero(self.extra < self.instance.capacity)

    def has_moves(self) -> bool:
        """Whether at least one ball can move between two different bins."""
        donors, receivers = self.donors(), self.receivers()
        if donors.size == 0 or receivers.size == 0:
            return False
        return not (donors.size == receivers.size == 1 and donors[0] == receivers[0])

    def move_delta(self, src, dst):
        """Profit change of moving one ball from ``src`` to ``dst`` (accepts index arrays)."""
        lower, extra = self.instance.lower, self.extra
        return (lower[dst] + extra[dst] + 1) - (lower[src] + extra[src])

    def apply_move(self, src: int, dst: int) -> None:
        self.gain += int(self.move_delta(src, dst))
        self.extra[src] -= 1
        self.extra[dst] += 1


# ---------------------------------------------------------------------------
# Construction, neighborhoods and local search
# ---------------------------------------------------------------------------


def greedy_solution(instance: Instance) -> Solution:
    """Fill bins up to their upper bound, largest upper bound first, until balls run out."""
    extra = np.zeros(instance.n_bins, dtype=np.int64)
    remaining = instance.free_balls
    for i in np.argsort(-instance.upper, kind="stable"):
        if remaining == 0:
            break
        extra[i] = min(instance.capacity[i], remaining)
        remaining -= extra[i]
    return Solution(instance, extra)


def shake(solution: Solution, k: int, rng: np.random.Generator) -> Solution:
    """Return a random point of the k-th neighborhood: ``k`` successive random ball moves."""
    neighbor = solution.copy()
    for _ in range(k):
        donors = neighbor.donors()
        if donors.size == 0:
            break
        src = rng.choice(donors)
        receivers = neighbor.receivers()
        receivers = receivers[receivers != src]
        if receivers.size == 0:
            break
        neighbor.apply_move(src, rng.choice(receivers))
    return neighbor


def sample_moves(
    solution: Solution, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample up to ``n_samples`` distinct single-ball moves ``src -> dst`` (``src != dst``)."""
    donors, receivers = solution.donors(), solution.receivers()
    if donors.size == 0 or receivers.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    src = rng.choice(donors, size=MAX_SAMPLING_TRIES)
    dst = rng.choice(receivers, size=MAX_SAMPLING_TRIES)
    valid = src != dst
    src, dst = src[valid], dst[valid]
    # Keep the first occurrence of each distinct move, in draw order.
    _, first = np.unique(src * solution.instance.n_bins + dst, return_index=True)
    keep = np.sort(first)[:n_samples]
    return src[keep], dst[keep]


def local_search(
    solution: Solution, rng: np.random.Generator, n_samples: int, max_iterations: int
) -> None:
    """Sampled best-improvement local search over single-ball moves (in place).

    Each iteration samples ``n_samples`` distinct moves and applies the best one
    if it improves the profit. The search stops at the first iteration whose
    sample has no improving move, or after ``max_iterations`` iterations.
    """
    for _ in range(max_iterations):
        src, dst = sample_moves(solution, n_samples, rng)
        if src.size == 0:
            return
        deltas = solution.move_delta(src, dst)
        best = int(np.argmax(deltas))
        if deltas[best] <= 0:
            return
        solution.apply_move(src[best], dst[best])


# ---------------------------------------------------------------------------
# Variable Neighborhood Search
# ---------------------------------------------------------------------------


@dataclass
class VNSConfig:
    max_neighborhood: int = 3      # initial largest neighborhood k_max (number of moves)
    neighborhood_limit: int = 7    # k_max never grows beyond this
    max_stagnation: int = 5        # sweeps without acceptance before k_max grows by one
    noise_level: float = 0.05      # std of the acceptance noise, relative to the current gain
    local_samples: int = 35        # moves sampled per local search iteration
    local_iterations: int = 25     # maximum local search iterations
    max_iterations: int = 1000     # maximum VNS sweeps (k = 1 .. k_max)
    time_limit: float = 3600.0     # seconds
    seed: int = 42


@dataclass
class VNSResult:
    balls_per_bin: list[int]
    profit: int
    initial_balls_per_bin: list[int]
    initial_profit: int
    iterations: int
    elapsed: float
    history: list[tuple[float, int]]  # (seconds, profit) each time the best improved


def vns(instance: Instance, config: VNSConfig | None = None, verbose: bool = False) -> VNSResult:
    """Maximize the profit of ``instance`` with VNS, starting from the greedy solution.

    One sweep tries neighborhoods ``k = 1 .. k_max``: shake ``k`` random moves,
    improve with local search, and accept the candidate when
    ``gain(candidate) + noise > gain(current)``, with zero-mean Gaussian noise of
    standard deviation ``noise_level * gain(current)``. Acceptance restarts the
    sweep at ``k = 1`` and shrinks ``k_max`` by one, never below its initial
    value; ``max_stagnation`` consecutive sweeps without acceptance grow
    ``k_max`` by one, up to ``neighborhood_limit``.
    """
    config = config or VNSConfig()
    start = time.perf_counter()
    rng = np.random.default_rng(config.seed)

    def elapsed() -> float:
        return time.perf_counter() - start

    initial = current = best = greedy_solution(instance)
    history = [(elapsed(), best.profit)]
    if verbose:
        print(f"[{history[0][0]:8.2f}s] greedy profit: {best.profit}", file=sys.stderr)

    k_max, stagnation, iteration = config.max_neighborhood, 0, 0
    # Moves keep the number of placed balls constant, so if no move exists the
    # greedy solution is the only candidate (every bin full, or nothing to place).
    searching = current.has_moves()

    while searching and iteration < config.max_iterations:
        iteration += 1
        accepted = False
        k = 1
        while k <= k_max:
            candidate = shake(current, k, rng)
            local_search(candidate, rng, config.local_samples, config.local_iterations)

            noise = config.noise_level * current.gain * rng.standard_normal()
            if candidate.gain + noise > current.gain:
                current, accepted, k = candidate, True, 1
                k_max = max(k_max - 1, config.max_neighborhood)
                if current.gain > best.gain:
                    best = current
                    history.append((elapsed(), best.profit))
                    if verbose:
                        print(f"[{history[-1][0]:8.2f}s] new best profit: {best.profit}",
                              file=sys.stderr)
            else:
                k += 1

            if elapsed() >= config.time_limit:
                searching = False
                break

        stagnation = 0 if accepted else stagnation + 1
        if stagnation >= config.max_stagnation:
            k_max = min(k_max + 1, config.neighborhood_limit)
            stagnation = 0

    return VNSResult(
        balls_per_bin=best.balls_per_bin(),
        profit=best.profit,
        initial_balls_per_bin=initial.balls_per_bin(),
        initial_profit=initial.profit,
        iterations=iteration,
        elapsed=elapsed(),
        history=history,
    )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    defaults = VNSConfig()
    p = argparse.ArgumentParser(
        description="Solve a Balls and Bins instance with Variable Neighborhood Search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("instance", type=Path, help="path to the instance file")
    p.add_argument("-s", "--seed", type=int, default=defaults.seed,
                   help="random seed")
    p.add_argument("-i", "--iterations", type=int, default=defaults.max_iterations,
                   help="maximum number of VNS sweeps")
    p.add_argument("-t", "--timeout", type=float, default=defaults.time_limit,
                   help="time limit in seconds")
    p.add_argument("-n", "--max-neighborhood", type=int, default=defaults.max_neighborhood,
                   help="initial largest neighborhood (number of random moves)")
    p.add_argument("-nl", "--neighborhood-limit", type=int, default=defaults.neighborhood_limit,
                   help="largest value the neighborhood size can grow to")
    p.add_argument("-g", "--max-stagnation", type=int, default=defaults.max_stagnation,
                   help="sweeps without acceptance before the neighborhood grows")
    p.add_argument("-x", "--noise-level", type=float, default=defaults.noise_level,
                   help="acceptance noise, relative to the current profit gain")
    p.add_argument("-ls", "--local-samples", type=int, default=defaults.local_samples,
                   help="moves sampled per local search iteration")
    p.add_argument("-li", "--local-iterations", type=int, default=defaults.local_iterations,
                   help="maximum local search iterations")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="write the run (configuration and results) as one JSON line to this file")
    p.add_argument("-a", "--append", action="store_true",
                   help="append to --output instead of overwriting it")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="do not log improvements while searching")
    args = p.parse_args(argv)

    if not args.instance.is_file():
        p.error(f"instance file not found: {args.instance}")
    if not 1 <= args.max_neighborhood <= args.neighborhood_limit:
        p.error("need 1 <= --max-neighborhood <= --neighborhood-limit")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    config = VNSConfig(
        max_neighborhood=args.max_neighborhood,
        neighborhood_limit=args.neighborhood_limit,
        max_stagnation=args.max_stagnation,
        noise_level=args.noise_level,
        local_samples=args.local_samples,
        local_iterations=args.local_iterations,
        max_iterations=args.iterations,
        time_limit=args.timeout,
        seed=args.seed,
    )
    try:
        instance = Instance.from_file(args.instance)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    result = vns(instance, config, verbose=not args.quiet)

    print(f"Instance:        {args.instance} ({instance.n_bins} bins, {instance.n_balls} balls)")
    print("Configuration:   " + ", ".join(f"{k}={v}" for k, v in asdict(config).items()))
    print(f"Initial profit:  {result.initial_profit} (greedy)")
    print(f"Best profit:     {result.profit}")
    print(f"Balls per bin:   {result.balls_per_bin}")
    print(f"Sweeps:          {result.iterations}")
    print(f"Time elapsed:    {result.elapsed:.2f} s")

    if args.output is not None:
        record = {"instance": str(args.instance), "config": asdict(config), **asdict(result)}
        with open(args.output, "a" if args.append else "w") as f:
            f.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
