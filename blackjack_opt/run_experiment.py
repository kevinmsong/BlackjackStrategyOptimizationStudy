"""
run_experiment.py — End-to-end experiment orchestrator.

Stages
  1  Solve oracle DP
  2  Validate oracle vs simulator
  3  Run optimizers  (PG · SPSA · CEM)
  4  Evaluate & build regret tables
  5  Bet-sizing study
  6  Export all figures → figures/
  7  Print summary table
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from blackjack_opt.config import BENCHMARK, ALL_VARIANTS, ACTION_NAMES
from blackjack_opt.env import BlackjackEnv
from blackjack_opt.oracle import solve_optimal_policy
from blackjack_opt.policy import Policy
from blackjack_opt.optim_pg import REINFORCEOptimizer
from blackjack_opt.optim_spsa import SPSAOptimizer
from blackjack_opt.optim_cem import CEMOptimizer
from blackjack_opt.betting import (
    prove_min_bet_optimal,
    simulate_bet_strategies,
    summarize_bet_strategies,
    BetOptimizer,
)
from blackjack_opt.evaluate import (
    evaluate_ev,
    compute_regret_table,
    action_match_rate,
    worst_cell_regret,
    summarize_comparison,
)
from blackjack_opt.export_tables import (
    export_strategy_charts,
    export_convergence_plot,
    export_regret_heatmap,
    export_comparison_csv,
    export_comparison_chart,
    export_bet_study_plot,
    export_all,
)

# ── Terminal formatting ───────────────────────────────────────────────────────

import io, sys
# Force UTF-8 on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

W = 68   # line width

def _hr(char="-"):
    print(char * W)

def _section(title: str):
    print()
    _hr("=")
    pad = (W - len(title) - 2) // 2
    print("=" * pad + "  " + title + "  " + "=" * (W - pad - len(title) - 2))
    _hr("=")

def _step(label: str):
    print(f"\n  >>  {label}")
    _hr("-")

def _ok(msg: str):
    print(f"      OK  {msg}")

def _info(msg: str):
    print(f"       -  {msg}")

def _warn(msg: str):
    print(f"    WARN  {msg}")

def _timing(label: str, elapsed: float):
    print(f"    time  {label}: {elapsed:.2f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blackjack Strategy Optimization — end-to-end experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--rules",       type=str, default="VEGAS_S17",
                   choices=list(ALL_VARIANTS.keys()))
    p.add_argument("--figures-dir", type=str, default="figures",
                   help="Output directory for all PNG figures")
    p.add_argument("--results-dir", type=str, default="results",
                   help="Output directory for CSV results")
    p.add_argument("--pg-hands",    type=int, default=1_000_000)
    p.add_argument("--spsa-iter",   type=int, default=8_000)
    p.add_argument("--cem-gen",     type=int, default=300)
    p.add_argument("--eval-hands",  type=int, default=100_000)
    p.add_argument("--no-pg",       action="store_true")
    p.add_argument("--no-spsa",     action="store_true")
    p.add_argument("--no-cem",      action="store_true")
    p.add_argument("--no-betting",  action="store_true")
    p.add_argument("--variants",    action="store_true",
                   help="Run EV sensitivity across all rule variants")
    p.add_argument("--quick",       action="store_true",
                   help="10 %% budget for fast sanity runs")
    return p.parse_args()


# ── Optimizer runners ─────────────────────────────────────────────────────────

def _pg_header(hands_total: int):
    print(f"\n  {'Hands':>10}  {'EV/hand':>9}  {'Entropy':>8}  {'Elapsed':>8}")
    _hr("·")

def _pg_row(metrics: dict):
    print(
        f"  {metrics['hands']:>10,}  "
        f"  {metrics['ev_per_hand']:>+8.4f}  "
        f"  {metrics['entropy']:>7.3f}  "
        f"  {metrics.get('elapsed_s', 0):>7.1f}s"
    )

def run_pg(env, oracle, args, seed) -> tuple[pd.DataFrame, dict]:
    _step(f"Policy Gradient / REINFORCE  (seed={seed}, hands={args.pg_hands:,})")
    cells   = list(oracle.Q_star.keys())
    policy  = Policy(BENCHMARK, cells)
    opt     = REINFORCEOptimizer(
        env=env, policy=policy,
        lr=3e-3, entropy_coef=0.05, entropy_anneal=0.99995,
        batch_size=64, seed=seed,
    )
    log_rows  = []
    t_start   = time.time()
    interval  = max(1_000, args.pg_hands // 60)

    _pg_header(args.pg_hands)
    while opt.total_hands < args.pg_hands:
        m = opt.update()
        if opt.total_hands % interval < opt.batch_size:
            m["elapsed_s"] = time.time() - t_start
            log_rows.append(m)
            _pg_row(m)

    elapsed = time.time() - t_start
    _timing("PG total", elapsed)
    return pd.DataFrame(log_rows), opt.policy.argmax_chart()


def _spsa_header():
    print(f"\n  {'Iter':>7}  {'Hands':>10}  {'EV/hand':>9}  {'‖∇‖':>8}  {'Elapsed':>8}")
    _hr("·")

def _spsa_row(m: dict):
    print(
        f"  {m['iteration']:>7,}  "
        f"  {m['hands']:>10,}  "
        f"  {m['ev_per_hand']:>+8.4f}  "
        f"  {m['grad_norm']:>7.4f}  "
        f"  {m.get('elapsed_s', 0):>7.1f}s"
    )

def run_spsa(env, oracle, args, seed) -> tuple[pd.DataFrame, dict]:
    _step(f"SPSA  (seed={seed}, iterations={args.spsa_iter:,})")
    cells  = list(oracle.Q_star.keys())
    policy = Policy(BENCHMARK, cells)
    opt    = SPSAOptimizer(
        env=env, policy=policy,
        a=0.5, c=0.2, alpha=0.602, gamma=0.101, A=100.0,
        n_eval=300, seed=seed,
    )
    log_rows = []
    t_start  = time.time()
    interval = max(10, args.spsa_iter // 40)

    _spsa_header()
    for _ in range(args.spsa_iter):
        m = opt.step()
        if opt.k % interval == 0:
            m["elapsed_s"] = time.time() - t_start
            log_rows.append(m)
            _spsa_row(m)

    _timing("SPSA total", time.time() - t_start)
    return pd.DataFrame(log_rows), opt.policy.argmax_chart()


def _cem_header():
    print(f"\n  {'Gen':>6}  {'Hands':>10}  {'Mean EV':>9}  {'Elite EV':>9}  {'σ mean':>7}  {'Elapsed':>8}")
    _hr("·")

def _cem_row(m: dict):
    print(
        f"  {m['generation']:>6,}  "
        f"  {m['hands']:>10,}  "
        f"  {m['ev_per_hand']:>+8.4f}  "
        f"  {m['elite_ev']:>+8.4f}  "
        f"  {m['std_mean']:>6.3f}  "
        f"  {m.get('elapsed_s', 0):>7.1f}s"
    )

def run_cem(env, oracle, args, seed) -> tuple[pd.DataFrame, dict]:
    _step(f"CEM  (seed={seed}, generations={args.cem_gen:,})")
    cells  = list(oracle.Q_star.keys())
    policy = Policy(BENCHMARK, cells)
    opt    = CEMOptimizer(
        env=env, policy=policy,
        n_samples=50, elite_frac=0.2, n_eval=500,
        noise_init=2.0, noise_decay=0.995,
        seed=seed,
    )
    log_rows = []
    t_start  = time.time()
    interval = max(1, args.cem_gen // 20)

    _cem_header()
    for _ in range(args.cem_gen):
        m = opt.step()
        if opt.generation % interval == 0:
            m["elapsed_s"] = time.time() - t_start
            log_rows.append(m)
            _cem_row(m)

    _timing("CEM total", time.time() - t_start)
    return pd.DataFrame(log_rows), opt.policy.argmax_chart()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.quick:
        args.pg_hands   = max(10_000, args.pg_hands   // 10)
        args.spsa_iter  = max(100,    args.spsa_iter   // 10)
        args.cem_gen    = max(20,     args.cem_gen     // 10)
        args.eval_hands = max(5_000,  args.eval_hands  // 10)

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    rules = ALL_VARIANTS[args.rules]
    rng   = np.random.default_rng(args.seed)
    env   = BlackjackEnv(rules, rng)

    _section("BLACKJACK STRATEGY OPTIMIZATION STUDY")
    _info(f"Rule set  : {rules.variant.name}")
    _info(f"Dealer    : {'S17' if rules.dealer_stands_soft17 else 'H17'}  "
          f"| 3:2 BJ  | Peek={rules.dealer_peek}  "
          f"| DAS={rules.double_after_split}  "
          f"| Surrender={rules.surrender_allowed}")
    _info(f"Shoe      : Infinite (i.i.d. draws)")
    _info(f"Seed      : {args.seed}")
    _info(f"Figures   : {os.path.abspath(args.figures_dir)}/")
    _info(f"Results   : {os.path.abspath(args.results_dir)}/")
    if args.quick:
        _warn("Quick mode active — training budgets at 10 %")

    # ── 1. Oracle ─────────────────────────────────────────────────────────────
    _section("STAGE 1 · ORACLE DP SOLVER")
    _step("Solving exact DP (infinite-shoe, conditioned on no dealer BJ under peek)")
    t0 = time.time()
    oracle = solve_optimal_policy(rules)
    _timing("Oracle solve", time.time() - t0)
    _ok(f"{len(oracle.pi_star):,} decision cells enumerated")

    _step("Exporting oracle strategy charts")
    export_strategy_charts(oracle.pi_star, rules, args.figures_dir, prefix="oracle")
    _ok("Hard · Soft · Pair charts saved  (oracle prefix)")

    # ── 2. Validation ─────────────────────────────────────────────────────────
    _section("STAGE 2 · ORACLE VALIDATION")
    _step(f"Monte Carlo simulation of oracle policy  ({args.eval_hands:,} hands)")
    t0 = time.time()
    sim_ev = evaluate_ev(env, oracle.pi_star, args.eval_hands, rng)
    _timing("Simulation", time.time() - t0)
    _ok(f"Simulated EV under oracle policy : {sim_ev:+.5f}")
    _info(f"Published benchmark (S17, ∞-shoe): ≈ −0.00460")
    gap = abs(sim_ev - (-0.00460))
    if gap > 0.010:
        _warn(f"Gap from published = {gap:.5f}  (>{0.010:.3f} — check oracle or n_hands)")
    else:
        _ok(f"Gap from published = {gap:.5f}  (within sampling tolerance)")

    # ── 3. Optimizers ─────────────────────────────────────────────────────────
    _section("STAGE 3 · OPTIMIZER RECOVERY")
    _info("Each optimizer attempts to recover the oracle policy from interaction alone.")

    optimizer_results: dict[str, dict] = {}
    logs:          dict[str, pd.DataFrame] = {}
    eval_policies: dict[str, dict]         = {}

    if not args.no_pg:
        log, pol = run_pg(env, oracle, args, seed=args.seed)
        rdf = compute_regret_table(pol, oracle, rules)
        optimizer_results["PG"] = {"log": log, "policy": pol, "regret": rdf}
        logs["PG"] = log
        eval_policies["PG"] = pol

    if not args.no_spsa:
        log, pol = run_spsa(env, oracle, args, seed=args.seed)
        rdf = compute_regret_table(pol, oracle, rules)
        optimizer_results["SPSA"] = {"log": log, "policy": pol, "regret": rdf}
        logs["SPSA"] = log
        eval_policies["SPSA"] = pol

    if not args.no_cem:
        log, pol = run_cem(env, oracle, args, seed=args.seed)
        rdf = compute_regret_table(pol, oracle, rules)
        optimizer_results["CEM"] = {"log": log, "policy": pol, "regret": rdf}
        logs["CEM"] = log
        eval_policies["CEM"] = pol

    # ── 4. Evaluation ─────────────────────────────────────────────────────────
    _section("STAGE 4 · EVALUATION & REGRET ANALYSIS")

    if eval_policies:
        comparison = summarize_comparison(
            logs, sim_ev, oracle.pi_star, eval_policies, oracle
        )
        _step("Comparison table")
        _hr()
        print(comparison.to_string(index=False))
        _hr()

        export_comparison_csv(comparison, os.path.join(args.results_dir, "comparison.csv"))
        export_comparison_chart(
            comparison, sim_ev,
            os.path.join(args.figures_dir, "optimizer_comparison.png"),
        )
        _ok("comparison.csv + optimizer_comparison.png saved")

        for opt_name, res in optimizer_results.items():
            rdf = res.get("regret")
            if rdf is not None and not rdf.empty:
                worst = worst_cell_regret(rdf, top_k=5)
                _step(f"Top-5 worst-regret cells  [{opt_name}]")
                print(worst[["cell_str", "optimal_name", "taken_name", "regret"]].to_string(index=False))

        optimizer_results["_oracle_ev"] = sim_ev
        _step("Exporting strategy charts, convergence plot, regret heatmaps")
        export_all(oracle.pi_star, optimizer_results, rules, args.figures_dir)
        if logs:
            export_convergence_plot(
                logs, sim_ev,
                os.path.join(args.figures_dir, "convergence.png"),
            )
        _ok("All figures exported to figures/")

    # ── 5. Bet sizing ──────────────────────────────────────────────────────────
    if not args.no_betting:
        _section("STAGE 5 · BET-SIZING STUDY  (negative control)")

        proof = prove_min_bet_optimal(sim_ev)
        proof_path = os.path.join(args.results_dir, "bet_sizing_proof.txt")
        with open(proof_path, "w") as f:
            f.write(proof)
        print(proof)

        _step("Simulating fixed bet strategies")
        n_bet_hands = min(5_000, args.eval_hands // 5)
        bet_df = simulate_bet_strategies(
            env, oracle.pi_star,
            min_bet=1.0, max_bet=100.0, bankroll=10_000.0,
            n_hands=n_bet_hands, n_trials=30,
            rng=np.random.default_rng(args.seed + 1),
        )
        summary_bet = summarize_bet_strategies(bet_df)
        _hr()
        print(summary_bet.to_string(index=False))
        _hr()
        bet_df.to_csv(os.path.join(args.results_dir, "bet_strategies.csv"), index=False)
        summary_bet.to_csv(os.path.join(args.results_dir, "bet_strategies_summary.csv"), index=False)

        _step("Running bet optimizer (grid search over bet sizes)")
        bet_opt = BetOptimizer(
            env, oracle.pi_star,
            min_bet=1.0, max_bet=100.0,
            n_eval=min(2_000, args.eval_hands // 5),
            rng=np.random.default_rng(args.seed + 2),
        )
        bet_result = bet_opt.optimize()
        _ok(f"Optimizer selected bet = {bet_result['best_bet']:.1f}  "
            f"(min = {bet_result['min_bet']:.1f})  →  {bet_result['conclusion']}")
        export_bet_study_plot(bet_result, os.path.join(args.figures_dir, "bet_study.png"))
        _ok("bet_study.png saved")

    # ── 6. Variant sensitivity ────────────────────────────────────────────────
    if args.variants:
        _section("STAGE 6 · RULE VARIANT SENSITIVITY")
        rows = []
        for vname, vrules in ALL_VARIANTS.items():
            _step(f"Variant: {vname}")
            voracle = solve_optimal_policy(vrules)
            venv    = BlackjackEnv(vrules, np.random.default_rng(args.seed))
            vev     = evaluate_ev(venv, voracle.pi_star, args.eval_hands,
                                  np.random.default_rng(args.seed))
            export_strategy_charts(voracle.pi_star, vrules, args.figures_dir, prefix=vname)
            rows.append({"variant": vname, "ev": vev,
                         "ev_vs_benchmark": vev - sim_ev})
            _ok(f"{vname:20s}  EV = {vev:+.5f}  Δ = {vev - sim_ev:+.5f}")

        var_df = pd.DataFrame(rows)
        var_df.to_csv(os.path.join(args.results_dir, "variant_evs.csv"), index=False)

    # ── Done ──────────────────────────────────────────────────────────────────
    _section("COMPLETE")
    _ok(f"Figures → {os.path.abspath(args.figures_dir)}/")
    _ok(f"Results → {os.path.abspath(args.results_dir)}/")
    _hr("═")
    print()


if __name__ == "__main__":
    main()
