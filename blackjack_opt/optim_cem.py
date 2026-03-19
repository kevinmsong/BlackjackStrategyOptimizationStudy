"""
optim_cem.py — Cross-Entropy Method over the space of action logit tables.

Each generation: sample N policies from N(mean_theta, std_theta),
evaluate each, keep elites, refit distribution.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from blackjack_opt.config import Rules
from blackjack_opt.env import BlackjackEnv
from blackjack_opt.policy import Policy


def _simulate_ev(
    policy: Policy, env: BlackjackEnv, n_hands: int, rng: np.random.Generator
) -> float:
    total = 0.0
    for _ in range(n_hands):
        state, cell = env.reset(bet=1.0)
        while not state.done:
            if cell is None:
                break
            action = policy.get_action_with_rng(cell, rng)
            state, cell, reward, done = env.step(state, action)
        total += state.reward
    return total / n_hands


class CEMOptimizer:
    """
    Cross-Entropy Method optimizer for blackjack policy logits.
    """

    def __init__(
        self,
        env: BlackjackEnv,
        policy: Policy,
        n_samples: int = 50,
        elite_frac: float = 0.2,
        n_eval: int = 500,
        noise_init: float = 2.0,
        noise_min: float = 0.05,
        noise_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        self.env          = env
        self.policy       = policy.clone()
        self.n_samples    = n_samples
        self.elite_frac   = elite_frac
        self.n_eval       = n_eval
        self.noise_min    = noise_min
        self.noise_decay  = noise_decay
        self.rng          = np.random.default_rng(seed)

        n = policy.n_cells * policy.theta.shape[1]
        self.mean_theta = policy.to_array()
        self.std_theta  = np.full(n, noise_init)

        self.generation  = 0
        self.total_hands = 0

    def _sample_population(self) -> list[Policy]:
        pops = []
        for _ in range(self.n_samples):
            theta_flat = self.rng.normal(self.mean_theta, self.std_theta)
            pops.append(self.policy.from_array(theta_flat))
        return pops

    def step(self) -> dict:
        population = self._sample_population()

        evs = []
        for pol in population:
            ev = _simulate_ev(pol, self.env, self.n_eval, self.rng)
            evs.append(ev)
        self.total_hands += self.n_samples * self.n_eval

        evs_arr = np.array(evs)
        n_elite = max(1, int(self.n_samples * self.elite_frac))
        elite_idx = np.argsort(evs_arr)[-n_elite:]

        elite_thetas = np.stack([population[i].to_array() for i in elite_idx])
        self.mean_theta = elite_thetas.mean(axis=0)
        self.std_theta  = elite_thetas.std(axis=0)
        self.std_theta  = np.maximum(self.std_theta, self.noise_min)
        self.std_theta  *= self.noise_decay

        # Update base policy to elite mean
        self.policy.theta = self.mean_theta.reshape(
            self.policy.n_cells, self.policy.theta.shape[1]
        )

        self.generation += 1
        return {
            "generation":  self.generation,
            "hands":       self.total_hands,
            "ev_per_hand": float(evs_arr.mean()),
            "elite_ev":    float(evs_arr[elite_idx].mean()),
            "best_ev":     float(evs_arr.max()),
            "std_mean":    float(self.std_theta.mean()),
        }

    def train(
        self,
        n_generations: int,
        eval_interval: int = 10,
        verbose: bool = True,
    ) -> pd.DataFrame:
        log_rows = []
        t_start  = time.time()

        for _ in range(n_generations):
            metrics = self.step()
            if self.generation % eval_interval == 0:
                metrics["elapsed_s"] = time.time() - t_start
                log_rows.append(metrics)
                if verbose:
                    print(
                        f"CEM  gen={self.generation:>5,}  "
                        f"hands={self.total_hands:>8,}  "
                        f"ev={metrics['ev_per_hand']:+.4f}  "
                        f"elite_ev={metrics['elite_ev']:+.4f}  "
                        f"elapsed={metrics['elapsed_s']:.1f}s"
                    )

        return pd.DataFrame(log_rows)
