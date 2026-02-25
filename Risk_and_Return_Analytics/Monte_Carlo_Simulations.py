"""
Monte Carlo Simulations for Portfolio Return Paths

Purpose
-------
Simulate portfolio value paths under a simple return-generating process to visualize
uncertainty, downside risk, and terminal wealth distributions.

This script is designed as a portfolio artifact to demonstrate:
- simulation-based risk analysis
- portfolio path modeling
- summary risk metrics
- clean, reproducible Python workflow

Notes
-----
- Uses a simple geometric return framework.
- By default, uses synthetic daily returns calibrated to annual assumptions.
- Can be extended to bootstrap historical returns instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class SimulationConfig:
    initial_portfolio_value: float = 100_000.0
    annual_expected_return: float = 0.10      # 10%
    annual_volatility: float = 0.18           # 18%
    trading_days_per_year: int = 252
    years: int = 3
    n_sims: int = 1000
    random_seed: int = 42
    downside_threshold_terminal: float = 90_000.0  # threshold for "loss" style probability metric


def annual_to_daily_params(annual_return: float, annual_vol: float, trading_days: int) -> Tuple[float, float]:
    """
    Convert annualized return/vol assumptions to daily parameters.

    We use a simple approximation for mean and volatility:
    - daily mean ~= annual_return / trading_days
    - daily vol  ~= annual_vol / sqrt(trading_days)
    """
    daily_mean = annual_return / trading_days
    daily_vol = annual_vol / np.sqrt(trading_days)
    return daily_mean, daily_vol


def simulate_portfolio_paths(cfg: SimulationConfig) -> pd.DataFrame:
    """
    Simulate geometric portfolio paths from daily returns.

    Returns
    -------
    DataFrame
        Shape: (n_days + 1, n_sims)
        Rows are time points, columns are simulation paths.
    """
    rng = np.random.default_rng(cfg.random_seed)
    n_days = cfg.trading_days_per_year * cfg.years

    daily_mean, daily_vol = annual_to_daily_params(
        cfg.annual_expected_return,
        cfg.annual_volatility,
        cfg.trading_days_per_year,
    )

    # Simulate daily arithmetic returns
    # Shape: (n_days, n_sims)
    daily_returns = rng.normal(loc=daily_mean, scale=daily_vol, size=(n_days, cfg.n_sims))

    # Build portfolio paths
    paths = np.zeros((n_days + 1, cfg.n_sims), dtype=float)
    paths[0, :] = cfg.initial_portfolio_value

    for t in range(1, n_days + 1):
        paths[t, :] = paths[t - 1, :] * (1.0 + daily_returns[t - 1, :])

    index = pd.RangeIndex(start=0, stop=n_days + 1, step=1, name="day")
    columns = [f"sim_{i+1}" for i in range(cfg.n_sims)]
    return pd.DataFrame(paths, index=index, columns=columns)


def summarize_terminal_distribution(paths_df: pd.DataFrame, cfg: SimulationConfig) -> Dict[str, float]:
    """
    Compute terminal wealth summary metrics.
    """
    terminal_values = paths_df.iloc[-1]
    initial = cfg.initial_portfolio_value

    terminal_returns = (terminal_values / initial) - 1.0

    summary = {
        "initial_portfolio_value": initial,
        "n_sims": float(cfg.n_sims),
        "horizon_years": float(cfg.years),
        "terminal_mean": float(terminal_values.mean()),
        "terminal_median": float(terminal_values.median()),
        "terminal_std": float(terminal_values.std(ddof=1)),
        "terminal_p05": float(np.percentile(terminal_values, 5)),
        "terminal_p25": float(np.percentile(terminal_values, 25)),
        "terminal_p75": float(np.percentile(terminal_values, 75)),
        "terminal_p95": float(np.percentile(terminal_values, 95)),
        "prob_terminal_below_initial": float((terminal_values < initial).mean()),
        "prob_terminal_below_threshold": float((terminal_values < cfg.downside_threshold_terminal).mean()),
        "mean_terminal_return": float(terminal_returns.mean()),
        "median_terminal_return": float(np.median(terminal_returns)),
    }
    return summary


def compute_pathwise_drawdowns(paths_df: pd.DataFrame) -> pd.Series:
    """
    Compute maximum drawdown for each simulation path.
    Returns a Series indexed by simulation column.
    """
    running_max = paths_df.cummax()
    drawdowns = (paths_df / running_max) - 1.0
    max_drawdown_by_path = drawdowns.min(axis=0)
    return max_drawdown_by_path


def save_outputs(
    paths_df: pd.DataFrame,
    summary: Dict[str, float],
    max_dd: pd.Series,
    output_dir: Path,
) -> None:
    """
    Save charts and summary tables for portfolio/recruiter-friendly review.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary to CSV
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "monte_carlo_summary.csv", index=False)

    # Save max drawdowns to CSV
    max_dd.rename("max_drawdown").to_csv(output_dir / "pathwise_max_drawdowns.csv", index=True)

    # Plot 1: Sample simulated paths
    plt.figure(figsize=(12, 6))
    n_plot = min(100, paths_df.shape[1])
    plt.plot(paths_df.iloc[:, :n_plot], alpha=0.25)
    plt.title("Monte Carlo Simulated Portfolio Paths")
    plt.xlabel("Trading Day")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.savefig(output_dir / "simulated_paths.png", dpi=150)
    plt.close()

    # Plot 2: Terminal distribution histogram
    plt.figure(figsize=(12, 6))
    plt.hist(paths_df.iloc[-1], bins=40)
    plt.title("Terminal Portfolio Value Distribution")
    plt.xlabel("Terminal Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "terminal_distribution.png", dpi=150)
    plt.close()

    # Plot 3: Max drawdown distribution
    plt.figure(figsize=(12, 6))
    plt.hist(max_dd.values, bins=40)
    plt.title("Distribution of Pathwise Maximum Drawdowns")
    plt.xlabel("Maximum Drawdown")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "max_drawdown_distribution.png", dpi=150)
    plt.close()


def print_summary(summary: Dict[str, float], max_dd: pd.Series) -> None:
    """
    Print a clean console summary for quick review.
    """
    print("=" * 70)
    print("MONTE CARLO PORTFOLIO SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Initial Portfolio Value:      ${summary['initial_portfolio_value']:,.2f}")
    print(f"Simulation Count:             {int(summary['n_sims'])}")
    print(f"Horizon (Years):              {int(summary['horizon_years'])}")
    print("-" * 70)
    print(f"Terminal Mean:                ${summary['terminal_mean']:,.2f}")
    print(f"Terminal Median:              ${summary['terminal_median']:,.2f}")
    print(f"Terminal Std Dev:             ${summary['terminal_std']:,.2f}")
    print(f"5th Percentile Terminal:      ${summary['terminal_p05']:,.2f}")
    print(f"95th Percentile Terminal:     ${summary['terminal_p95']:,.2f}")
    print("-" * 70)
    print(f"Mean Terminal Return:         {summary['mean_terminal_return']:.2%}")
    print(f"Median Terminal Return:       {summary['median_terminal_return']:.2%}")
    print(f"P(Terminal < Initial):        {summary['prob_terminal_below_initial']:.2%}")
    print(f"P(Terminal < Threshold):      {summary['prob_terminal_below_threshold']:.2%}")
    print("-" * 70)
    print(f"Avg Max Drawdown (Pathwise):  {max_dd.mean():.2%}")
    print(f"Median Max Drawdown:          {max_dd.median():.2%}")
    print(f"Worst Max Drawdown (Pathwise):{max_dd.min():.2%}")
    print("=" * 70)


def main() -> None:
    cfg = SimulationConfig()

    paths_df = simulate_portfolio_paths(cfg)
    summary = summarize_terminal_distribution(paths_df, cfg)
    max_dd = compute_pathwise_drawdowns(paths_df)

    # Save outputs in a local folder for screenshots / portfolio assets
    output_dir = Path(__file__).resolve().parent / "outputs"
    save_outputs(paths_df, summary, max_dd, output_dir)

    print_summary(summary, max_dd)
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
