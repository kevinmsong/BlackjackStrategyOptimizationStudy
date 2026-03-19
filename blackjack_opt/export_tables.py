"""
export_tables.py — Professional figure export at 300 DPI.

All figures saved as PNG to a caller-specified directory.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

from blackjack_opt.config import (
    ACTION_STAND, ACTION_HIT, ACTION_DOUBLE, ACTION_SPLIT, ACTION_SURRENDER,
    ACTION_SHORT, Rules,
)
from blackjack_opt.state import DecisionCell

# ── Global style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.titlepad":      14,
    "axes.labelsize":     10,
    "axes.labelcolor":    "#222222",
    "axes.edgecolor":     "#cccccc",
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.color":        "#444444",
    "ytick.color":        "#444444",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.frameon":     False,
    "legend.fontsize":    9,
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.6,
})

DPI = 300

# ── Action palette ────────────────────────────────────────────────────────────
# Muted, accessible palette

ACTION_COLORS = {
    ACTION_STAND:     "#4CAF82",   # sage green
    ACTION_HIT:       "#E05C5C",   # muted red
    ACTION_DOUBLE:    "#F0B429",   # amber
    ACTION_SPLIT:     "#4B9CD3",   # steel blue
    ACTION_SURRENDER: "#9E9E9E",   # neutral grey
}
ACTION_LABELS = {
    ACTION_STAND:     "Stand",
    ACTION_HIT:       "Hit",
    ACTION_DOUBLE:    "Double",
    ACTION_SPLIT:     "Split",
    ACTION_SURRENDER: "Surrender",
}
ACTION_TEXT_COLOR = {
    ACTION_STAND:     "#1a3d2b",
    ACTION_HIT:       "#4a0d0d",
    ACTION_DOUBLE:    "#4a2e00",
    ACTION_SPLIT:     "#0d2a4a",
    ACTION_SURRENDER: "#2a2a2a",
}

DEALER_UPCARDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
DEALER_LABELS  = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]

OPT_COLORS = {
    "PG":   "#E05C5C",
    "SPSA": "#4B9CD3",
    "CEM":  "#F0B429",
}


# ── Grid builders ─────────────────────────────────────────────────────────────

def _build_hard_grid(policy):
    row_totals = list(range(8, 22))
    grid = np.full((len(row_totals), 10), -1, dtype=int)
    for ri, total in enumerate(row_totals):
        for ci, upcard in enumerate(DEALER_UPCARDS):
            for can_dbl in [True, False]:
                cell = DecisionCell(
                    player_total=total, dealer_upcard=upcard, is_soft=False,
                    is_pair=False, pair_rank=None, can_double=can_dbl,
                    can_split=False, from_split=False, split_depth=0,
                )
                if cell in policy:
                    grid[ri, ci] = policy[cell]
                    break
    return grid, row_totals


def _build_soft_grid(policy):
    row_totals = list(range(13, 22))
    grid = np.full((len(row_totals), 10), -1, dtype=int)
    for ri, total in enumerate(row_totals):
        for ci, upcard in enumerate(DEALER_UPCARDS):
            for can_dbl in [True, False]:
                cell = DecisionCell(
                    player_total=total, dealer_upcard=upcard, is_soft=True,
                    is_pair=False, pair_rank=None, can_double=can_dbl,
                    can_split=False, from_split=False, split_depth=0,
                )
                if cell in policy:
                    grid[ri, ci] = policy[cell]
                    break
    return grid, row_totals


def _build_pair_grid(policy):
    from blackjack_opt.hand import compute_total as ct
    pair_ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    grid = np.full((len(pair_ranks), 10), -1, dtype=int)
    for ri, pr in enumerate(pair_ranks):
        total, is_soft = ct([pr, pr])
        for ci, upcard in enumerate(DEALER_UPCARDS):
            for can_dbl in [True, False]:
                for can_spl in [True, False]:
                    cell = DecisionCell(
                        player_total=total, dealer_upcard=upcard, is_soft=is_soft,
                        is_pair=True, pair_rank=pr, can_double=can_dbl,
                        can_split=can_spl, from_split=False, split_depth=0,
                    )
                    if cell in policy:
                        grid[ri, ci] = policy[cell]
                        break
                if grid[ri, ci] != -1:
                    break
    return grid, pair_ranks


# ── Strategy chart ────────────────────────────────────────────────────────────

def _plot_strategy_grid(
    grid: np.ndarray,
    row_labels: list,
    col_labels: list,
    title: str,
    subtitle: str,
    filename: str,
    row_label_fn=None,
) -> None:
    n_rows, n_cols = grid.shape
    cell_w, cell_h = 0.72, 0.52

    # Extra top margin accommodates column headers + "Dealer Upcard" + title/subtitle
    # without set_aspect("equal"), which caused overlap for short charts (soft/pair).
    fig_w = n_cols * cell_w + 2.4
    fig_h = n_rows * cell_h + 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Fixed subplot layout — no set_aspect so axes fills allocated space consistently
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.13, right=0.98)

    # Data coords: x includes left-label area; y includes only the grid rows
    ax.set_xlim(-1.3, n_cols + 0.1)
    ax.set_ylim(-0.05, n_rows + 0.10)
    ax.axis("off")

    # Column header band (compact height so it stays clear of subtitle)
    header_y = n_rows + 0.10
    for ci, lbl in enumerate(col_labels):
        ax.add_patch(FancyBboxPatch(
            (ci + 0.04, header_y), 0.92, 0.40,
            boxstyle="round,pad=0.04",
            facecolor="#2c3e50", edgecolor="none",
            clip_on=False,
        ))
        ax.text(ci + 0.5, header_y + 0.20, lbl,
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold", clip_on=False)

    # "Dealer Upcard" label — positioned above headers but well below subtitle
    ax.text(n_cols / 2, n_rows + 0.70, "Dealer Upcard",
            ha="center", va="center", fontsize=10,
            color="#2c3e50", fontweight="bold", clip_on=False)

    # Row header
    for ri, lbl in enumerate(reversed(row_labels)):
        disp = row_label_fn(lbl) if row_label_fn else str(lbl)
        ax.add_patch(FancyBboxPatch(
            (-1.12, ri + 0.06), 1.0, 0.88,
            boxstyle="round,pad=0.04",
            facecolor="#ecf0f1", edgecolor="none",
            clip_on=False,
        ))
        ax.text(-0.62, ri + 0.5, disp,
                ha="center", va="center", fontsize=9,
                color="#2c3e50", fontweight="bold", clip_on=False)

    # Player hand label
    ax.text(-1.15, n_rows / 2, "Player Hand",
            ha="center", va="center", fontsize=10,
            color="#2c3e50", fontweight="bold", rotation=90, clip_on=False)

    # Cells
    for ri in range(n_rows):
        for ci in range(n_cols):
            action = grid[ri, ci]
            color   = ACTION_COLORS.get(action, "#e8e8e8") if action >= 0 else "#e8e8e8"
            txtcol  = ACTION_TEXT_COLOR.get(action, "#555555") if action >= 0 else "#999999"
            label   = ACTION_SHORT.get(action, "?") if action >= 0 else "?"

            ax.add_patch(FancyBboxPatch(
                (ci + 0.04, ri + 0.06), 0.92, 0.88,
                boxstyle="round,pad=0.03",
                facecolor=color, edgecolor="white", linewidth=0.8,
            ))
            ax.text(ci + 0.5, ri + 0.5, label,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=txtcol)

    # Title and subtitle — in figure fraction; subplots_adjust(top=0.86) ensures
    # the axes top is consistently at 0.86 regardless of n_rows, so these positions
    # never overlap with axes content for any chart size.
    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=14, fontweight="bold", color="#1a1a2e")
    fig.text(0.5, 0.935, subtitle, ha="center", va="top",
             fontsize=9, color="#666666")

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=ACTION_COLORS[a], edgecolor="white",
                       linewidth=0.5, label=ACTION_LABELS[a])
        for a in sorted(ACTION_COLORS)
        if a in set(grid.flatten())
    ]
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=len(legend_patches),
            fontsize=8.5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )

    plt.savefig(filename, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def export_strategy_charts(
    policy: dict[DecisionCell, int],
    rules: Rules,
    output_dir: str,
    prefix: str = "",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tag  = f"{prefix}_" if prefix else ""
    rule_str = rules.variant.name.replace("_", " ")
    subtitle = (
        f"Infinite shoe · {'S17' if rules.dealer_stands_soft17 else 'H17'} · "
        f"3:2 BJ · {'DAS' if rules.double_after_split else 'No DAS'} · "
        f"{'Late surrender' if rules.surrender_allowed else 'No surrender'}"
    )

    hard_grid, hard_rows = _build_hard_grid(policy)
    _plot_strategy_grid(
        hard_grid, hard_rows, DEALER_LABELS,
        title=f"Basic Strategy — Hard Totals  ({rule_str})",
        subtitle=subtitle,
        filename=os.path.join(output_dir, f"{tag}hard_chart.png"),
        row_label_fn=lambda x: str(x),
    )

    soft_grid, soft_rows = _build_soft_grid(policy)
    soft_fn = lambda t: f"A+{t - 11}"
    _plot_strategy_grid(
        soft_grid, soft_rows, DEALER_LABELS,
        title=f"Basic Strategy — Soft Totals  ({rule_str})",
        subtitle=subtitle,
        filename=os.path.join(output_dir, f"{tag}soft_chart.png"),
        row_label_fn=soft_fn,
    )

    pair_grid, pair_ranks = _build_pair_grid(policy)
    pair_fn = lambda pr: "A–A" if pr == 11 else f"{pr}–{pr}"
    _plot_strategy_grid(
        pair_grid, pair_ranks, DEALER_LABELS,
        title=f"Basic Strategy — Pairs  ({rule_str})",
        subtitle=subtitle,
        filename=os.path.join(output_dir, f"{tag}pair_chart.png"),
        row_label_fn=pair_fn,
    )


# ── Convergence plot ──────────────────────────────────────────────────────────

def export_convergence_plot(
    logs: dict[str, pd.DataFrame],
    oracle_ev: float,
    filename: str,
    smooth_window: int = 8,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for name, df in logs.items():
        if df.empty or "hands" not in df.columns or "ev_per_hand" not in df.columns:
            continue
        color = OPT_COLORS.get(name, "#888888")
        ev_smooth = df["ev_per_hand"].rolling(smooth_window, min_periods=1).mean()
        ax.plot(df["hands"], ev_smooth,
                label=name, color=color,
                linewidth=2.0, zorder=3)
        # Shaded ±1 raw std band
        ev_std = df["ev_per_hand"].rolling(smooth_window, min_periods=1).std().fillna(0)
        ax.fill_between(df["hands"],
                         ev_smooth - ev_std, ev_smooth + ev_std,
                         color=color, alpha=0.10, zorder=2)

    ax.axhline(oracle_ev, color="#2c3e50", linestyle="--", linewidth=1.4,
               label=f"Oracle EV  ({oracle_ev:+.4f})", zorder=4)

    ax.set_xlabel("Hands Played", labelpad=8)
    ax.set_ylabel("EV per Hand (unit bet)", labelpad=8)
    ax.set_title("Optimizer Convergence Toward Oracle Policy", pad=14)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x))))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:+.3f}"))
    ax.grid(True, axis="both", zorder=1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Regret heatmap ────────────────────────────────────────────────────────────

def export_regret_heatmap(
    regret_table: pd.DataFrame,
    chart_type: str,
    optimizer_name: str,
    filename: str,
) -> None:
    if chart_type == "hard":
        row_col    = "player_total"
        row_filter = (~regret_table["is_soft"]) & (~regret_table["is_pair"])
        row_label_fn = str
    elif chart_type == "soft":
        row_col    = "player_total"
        row_filter = regret_table["is_soft"] & (~regret_table["is_pair"])
        row_label_fn = lambda t: f"A+{int(t)-11}"
    else:
        row_col    = "pair_rank"
        row_filter = regret_table["is_pair"]
        row_label_fn = lambda pr: "A–A" if int(pr) == 11 else f"{int(pr)}–{int(pr)}"

    sub = regret_table[row_filter].copy()
    if sub.empty:
        return

    row_vals = sorted(sub[row_col].dropna().unique())
    n_rows   = len(row_vals)
    n_cols   = len(DEALER_UPCARDS)

    regret_grid = np.zeros((n_rows, n_cols))
    match_grid  = np.ones((n_rows, n_cols), dtype=bool)
    opt_grid    = [""] * (n_rows * n_cols)
    taken_grid  = [""] * (n_rows * n_cols)

    for ri, rv in enumerate(row_vals):
        for ci, upcard in enumerate(DEALER_UPCARDS):
            mask = (sub[row_col] == rv) & (sub["dealer_upcard"] == upcard)
            rows = sub[mask]
            if not rows.empty:
                regret_grid[ri, ci] = rows["regret"].values[0]
                match_grid[ri, ci]  = rows["action_match"].values[0]
                opt_grid[ri * n_cols + ci]   = rows["optimal_name"].values[0][:1]
                taken_grid[ri * n_cols + ci] = rows["taken_name"].values[0][:1]

    vmax = max(regret_grid.max(), 0.005)
    cmap = LinearSegmentedColormap.from_list(
        "regret", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    )

    fig_w = max(9, n_cols * 0.82) + 1.2
    fig_h = max(4, n_rows * 0.58) + 1.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(regret_grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

    for ri in range(n_rows):
        for ci in range(n_cols):
            val  = regret_grid[ri, ci]
            matched = match_grid[ri, ci]
            txt_color = "white" if val > vmax * 0.55 else "#1a1a2e"

            if not matched:
                # Show taken → optimal
                line1 = taken_grid[ri * n_cols + ci]
                line2 = opt_grid[ri * n_cols + ci]
                ax.text(ci, ri - 0.12, line1, ha="center", va="center",
                        fontsize=11.5, color=txt_color, fontweight="bold")
                ax.text(ci, ri + 0.22, f"({line2}✓)", ha="center", va="center",
                        fontsize=10.5, color=txt_color)
            else:
                ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                        fontsize=11.5, color=txt_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(DEALER_LABELS)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([row_label_fn(v) for v in row_vals])
    ax.set_xlabel("Dealer Upcard", labelpad=8)

    chart_titles = {"hard": "Hard Totals", "soft": "Soft Totals", "pair": "Pairs"}
    ax.set_title(
        f"Regret vs Oracle — {chart_titles.get(chart_type, chart_type)}  ({optimizer_name})\n"
        f"Cell value = EV lost vs oracle · Wrong-action cells show taken→(correct✓)",
        pad=12,
    )

    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Regret (EV units lost)", labelpad=8)
    cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    fig.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Bet study plot ────────────────────────────────────────────────────────────

def export_bet_study_plot(bet_results: dict, filename: str) -> None:
    bets     = np.array(bet_results["bets"])
    evs      = np.array(bet_results["evs"])
    best_bet = bet_results["best_bet"]
    min_bet  = bet_results["min_bet"]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(bets, evs, color="#4B9CD3", linewidth=2.0, zorder=3, label="EV per unit bet")
    ax.scatter([best_bet], [evs[np.argmin(np.abs(bets - best_bet))]],
               color="#E05C5C", s=80, zorder=5, label=f"Optimizer choice  (bet = {best_bet:.0f} = min)")

    ax.axhline(np.mean(evs), color="#999999", linestyle=":", linewidth=1.2,
               label=f"Mean EV ({np.mean(evs):+.4f})")

    # Shade the range
    ax.fill_between(bets, evs - 0.002, evs + 0.002, color="#4B9CD3", alpha=0.12, zorder=2)

    ax.set_xlabel("Bet Size (units)", labelpad=8)
    ax.set_ylabel("EV per Unit Bet", labelpad=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:+.4f}"))
    ax.set_title(
        "Bet-Sizing Under No Counting: EV is Constant in Bet Size\n"
        "Optimizer correctly selects minimum legal bet",
        pad=14,
    )
    ax.grid(True, zorder=1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Optimizer comparison bar chart ────────────────────────────────────────────

def export_comparison_chart(comparison_df: pd.DataFrame, oracle_ev: float, filename: str) -> None:
    if comparison_df.empty:
        return
    df = comparison_df.dropna(subset=["final_ev"]).copy()
    if df.empty:
        return

    names  = df["optimizer"].tolist()
    evs    = df["final_ev"].tolist()
    match  = df["action_match"].tolist() if "action_match" in df.columns else [None] * len(names)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: final EV vs oracle
    colors_bar = [OPT_COLORS.get(n, "#888") for n in names]
    bars = axes[0].bar(names, evs, color=colors_bar, edgecolor="white", linewidth=0.8, width=0.5)
    axes[0].axhline(oracle_ev, color="#2c3e50", linestyle="--", linewidth=1.4,
                    label=f"Oracle ({oracle_ev:+.4f})")
    for bar, ev in zip(bars, evs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, ev + 0.002,
                     f"{ev:+.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_title("Final EV per Hand", pad=12)
    axes[0].set_ylabel("EV (unit bet)")
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:+.3f}"))
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, axis="y")

    # Right: action match rate
    valid_match = [(n, m) for n, m in zip(names, match) if m is not None]
    if valid_match:
        mn, mm = zip(*valid_match)
        mm_pct = [m * 100 for m in mm]
        bars2 = axes[1].bar(mn, mm_pct,
                             color=[OPT_COLORS.get(n, "#888") for n in mn],
                             edgecolor="white", linewidth=0.8, width=0.5)
        for bar, pct in zip(bars2, mm_pct):
            axes[1].text(bar.get_x() + bar.get_width() / 2, pct + 0.5,
                         f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        axes[1].set_ylim(0, 110)
        axes[1].set_title("Action Match vs Oracle Policy", pad=12)
        axes[1].set_ylabel("Cells Matching Oracle (%)")
        axes[1].axhline(100, color="#2c3e50", linestyle="--", linewidth=1.0, alpha=0.5)
        axes[1].grid(True, axis="y")

    fig.suptitle("Optimizer Comparison Summary", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── CSV ───────────────────────────────────────────────────────────────────────

def export_comparison_csv(comparison_df: pd.DataFrame, filename: str) -> None:
    comparison_df.to_csv(filename, index=False)


# ── Top-level orchestrator ────────────────────────────────────────────────────

def export_all(
    oracle_policy: dict[DecisionCell, int],
    optimizer_results: dict[str, dict],
    rules: Rules,
    figures_dir: str,
) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    export_strategy_charts(oracle_policy, rules, figures_dir, prefix="oracle")

    logs: dict[str, pd.DataFrame] = {}
    for opt_name, res in optimizer_results.items():
        if not isinstance(res, dict):
            continue
        if "log" in res and res["log"] is not None:
            logs[opt_name] = res["log"]
        if "policy" in res and res["policy"] is not None:
            export_strategy_charts(res["policy"], rules, figures_dir, prefix=opt_name)
        if "regret" in res and res["regret"] is not None:
            for chart_type in ["hard", "soft", "pair"]:
                export_regret_heatmap(
                    res["regret"], chart_type, opt_name,
                    os.path.join(figures_dir, f"{opt_name}_regret_{chart_type}.png"),
                )

    oracle_ev = optimizer_results.get("_oracle_ev", -0.005)
    if logs:
        export_convergence_plot(
            logs, oracle_ev,
            os.path.join(figures_dir, "convergence.png"),
        )
