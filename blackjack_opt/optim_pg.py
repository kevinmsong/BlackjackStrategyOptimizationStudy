"""
optim_pg.py — Masked REINFORCE with per-cell baseline and Adam optimizer.

Implements: ∇J(θ) ≈ Σ_t ∇log π_θ(a_t|c_t) (G_t - b(c_t))
with entropy bonus and per-cell EMA baseline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd

from blackjack_opt.config import Rules, NUM_ACTIONS
from blackjack_opt.env import BlackjackEnv
from blackjack_opt.policy import Policy, masked_softmax
from blackjack_opt.state import DecisionCell


def _adam_step(
    theta: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    theta = theta + lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v


class REINFORCEOptimizer:
    """
    REINFORCE policy gradient optimizer for blackjack.

    Training loop: collect trajectory → compute advantage → gradient step.
    Baseline per cell is an EMA of observed returns (dramatically reduces variance).
    """

    def __init__(
        self,
        env: BlackjackEnv,
        policy: Policy,
        lr: float = 3e-3,
        entropy_coef: float = 0.05,
        entropy_anneal: float = 0.99995,
        baseline_alpha: float = 0.02,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.env           = env
        self.policy        = policy
        self.lr            = lr
        self.entropy_coef  = entropy_coef
        self.entropy_anneal= entropy_anneal
        self.baseline_alpha= baseline_alpha
        self.batch_size    = batch_size
        self.rng           = np.random.default_rng(seed)

        # Adam state
        self.adam_m = np.zeros_like(policy.theta)
        self.adam_v = np.zeros_like(policy.theta)
        self.adam_t = 0

        # Per-cell baseline (EMA of returns)
        self.baselines: dict[DecisionCell, float] = {}

        # Metrics
        self.total_hands = 0

    def _collect_episode(self) -> tuple[list[tuple[int, int, DecisionCell]], float]:
        """
        Run one hand. Returns (trajectory, terminal_reward).
        trajectory: list of (cell_idx, action, reward_placeholder)
        """
        state, cell = self.env.reset(bet=1.0)
        trajectory: list[tuple[int, int]] = []   # (cell_idx, action)

        while not state.done:
            if cell is None:
                break
            idx = self.policy.cell_to_idx.get(cell)
            if idx is None:
                # Unknown cell: default stand
                action = 0
            else:
                probs = masked_softmax(
                    self.policy.theta[idx], self.policy.legal_masks[idx]
                )
                action = int(self.rng.choice(NUM_ACTIONS, p=probs))

            state, cell, reward, done = self.env.step(state, action)
            trajectory.append((idx, action))

        return trajectory, state.reward

    def _collect_batch(self) -> tuple[list, list[float]]:
        trajs, rewards = [], []
        for _ in range(self.batch_size):
            traj, r = self._collect_episode()
            trajs.append(traj)
            rewards.append(r)
        return trajs, rewards

    def update(self) -> dict:
        trajs, rewards = self._collect_batch()
        self.total_hands += self.batch_size

        grad = np.zeros_like(self.policy.theta)
        total_reward = 0.0
        total_entropy = 0.0
        n_steps = 0

        for traj, G in zip(trajs, rewards):
            total_reward += G
            for idx, action in traj:
                if idx is None:
                    continue
                cell = self.policy.cells[idx]
                # Update baseline
                b = self.baselines.get(cell, 0.0)
                b = (1 - self.baseline_alpha) * b + self.baseline_alpha * G
                self.baselines[cell] = b

                advantage = G - b

                # REINFORCE gradient: advantage * (e_action - probs)
                probs = masked_softmax(
                    self.policy.theta[idx], self.policy.legal_masks[idx]
                )
                e_action = np.zeros(NUM_ACTIONS)
                e_action[action] = 1.0
                pg_grad = advantage * (e_action - probs)

                # Entropy gradient: -log_pi - 1 (approx)
                log_probs = np.log(probs + 1e-30)
                ent_grad = -(log_probs + 1.0) * probs
                ent_grad[~self.policy.legal_masks[idx]] = 0.0

                total_entropy += float(-np.sum(probs * log_probs))

                grad[idx] += pg_grad + self.entropy_coef * ent_grad
                n_steps += 1

        if n_steps > 0:
            grad /= n_steps

        # Adam step (gradient ascent: maximize J)
        self.adam_t += 1
        self.policy.theta, self.adam_m, self.adam_v = _adam_step(
            self.policy.theta, grad,
            self.adam_m, self.adam_v,
            self.adam_t, self.lr,
        )

        # Anneal entropy coefficient
        self.entropy_coef *= self.entropy_anneal

        mean_reward  = total_reward / self.batch_size
        mean_entropy = total_entropy / max(n_steps, 1)
        grad_norm    = float(np.linalg.norm(grad))

        return {
            "hands":       self.total_hands,
            "ev_per_hand": mean_reward,
            "entropy":     mean_entropy,
            "grad_norm":   grad_norm,
            "entropy_coef":self.entropy_coef,
        }

    def train(
        self,
        total_hands: int,
        eval_interval: int = 10_000,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Full training loop. Returns convergence log DataFrame.
        """
        log_rows = []
        t_start  = time.time()
        hands_so_far = 0

        while hands_so_far < total_hands:
            metrics = self.update()
            hands_so_far = self.total_hands

            if hands_so_far % eval_interval < self.batch_size:
                elapsed = time.time() - t_start
                metrics["elapsed_s"] = elapsed
                log_rows.append(metrics)
                if verbose:
                    print(
                        f"PG  hands={hands_so_far:>8,}  "
                        f"ev={metrics['ev_per_hand']:+.4f}  "
                        f"ent={metrics['entropy']:.3f}  "
                        f"elapsed={elapsed:.1f}s"
                    )

        return pd.DataFrame(log_rows)
