"""
optim_spsa.py — Simultaneous Perturbation Stochastic Approximation.

Estimates the gradient from two function evaluations with ±δ perturbation
in a random Bernoulli direction, following Spall (1992) parameter schedules.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from blackjack_opt.config import Rules
from blackjack_opt.env import BlackjackEnv
from blackjack_opt.policy import Policy


def _simulate_ev(policy: Policy, env: BlackjackEnv, n_hands: int, rng: np.random.Generator) -> float:
    """Monte Carlo EV estimate for a policy."""
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


class SPSAOptimizer:
    """
    SPSA optimizer for blackjack policy logits.

    Step sizes follow Spall (1992) decay schedules:
      a_k = a / (k + 1 + A)^alpha
      c_k = c / (k + 1)^gamma
    """

    def __init__(
        self,
        env: BlackjackEnv,
        policy: Policy,
        a: float = 0.5,
        c: float = 0.2,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 100.0,
        n_eval: int = 300,
        seed: int = 42,
    ) -> None:
        self.env    = env
        self.policy = policy.clone()
        self.a      = a
        self.c      = c
        self.alpha  = alpha
        self.gamma  = gamma
        self.A      = A
        self.n_eval = n_eval
        self.rng    = np.random.default_rng(seed)
        self.k      = 0
        self.total_hands = 0

    def _ak(self) -> float:
        return self.a / (self.k + 1 + self.A) ** self.alpha

    def _ck(self) -> float:
        return self.c / (self.k + 1) ** self.gamma

    def step(self) -> dict:
        ak = self._ak()
        ck = self._ck()

        theta = self.policy.to_array()
        n = len(theta)

        # Bernoulli ±1 perturbation
        delta = (self.rng.integers(0, 2, size=n) * 2 - 1).astype(np.float64)

        theta_plus  = theta + ck * delta
        theta_minus = theta - ck * delta

        pol_plus  = self.policy.from_array(theta_plus)
        pol_minus = self.policy.from_array(theta_minus)

        ev_plus  = _simulate_ev(pol_plus,  self.env, self.n_eval, self.rng)
        ev_minus = _simulate_ev(pol_minus, self.env, self.n_eval, self.rng)

        # Gradient estimate (element-wise)
        grad_est = (ev_plus - ev_minus) / (2.0 * ck * delta)

        # Gradient ascent update
        new_theta = theta + ak * grad_est
        self.policy.theta = new_theta.reshape(self.policy.n_cells, -1)

        self.k += 1
        self.total_hands += 2 * self.n_eval

        return {
            "iteration":   self.k,
            "hands":       self.total_hands,
            "ev_plus":     ev_plus,
            "ev_minus":    ev_minus,
            "ev_per_hand": (ev_plus + ev_minus) / 2.0,
            "grad_norm":   float(np.linalg.norm(grad_est)),
            "ak":          ak,
            "ck":          ck,
        }

    def train(
        self,
        n_iterations: int,
        eval_interval: int = 100,
        verbose: bool = True,
    ) -> pd.DataFrame:
        log_rows = []
        t_start  = time.time()

        for _ in range(n_iterations):
            metrics = self.step()
            if self.k % eval_interval == 0:
                metrics["elapsed_s"] = time.time() - t_start
                log_rows.append(metrics)
                if verbose:
                    print(
                        f"SPSA iter={self.k:>6,}  "
                        f"hands={self.total_hands:>8,}  "
                        f"ev={metrics['ev_per_hand']:+.4f}  "
                        f"elapsed={metrics['elapsed_s']:.1f}s"
                    )

        return pd.DataFrame(log_rows)
