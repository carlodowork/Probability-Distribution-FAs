import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
FIT_SAMPLE_SIZE = 200_000
SHAPIRO_SAMPLE_SIZE = 5_000
DISTRIBUTIONS = {
    "norm": stats.norm,
    "t": stats.t,
    "laplace": stats.laplace,
    "logistic": stats.logistic,
    "cauchy": stats.cauchy,
}


class DataError(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="SA1 Section II: distribution fitting and normality testing")
    parser.add_argument("--btc", type=Path, help="Bitcoin 1-minute CSV file")
    parser.add_argument("--eth", type=Path, help="Ethereum 5-minute CSV file")
    parser.add_argument("--price-column", default="Close")
    parser.add_argument("--time-column", default="Timestamp")
    parser.add_argument("--btc-start", default="2012-01-01")
    parser.add_argument("--btc-end", default="2026-09-15")
    parser.add_argument("--eth-start", default="2015-08-07")
    parser.add_argument("--eth-end", default="2026-09-15")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def validate_args(args):
    if not 0 < args.alpha < 1:
        raise DataError("alpha must be strictly between 0 and 1")
    if args.btc is None and args.eth is None:
        raise DataError("provide at least one of --btc or --eth")
    for label, path in (("btc", args.btc), ("eth", args.eth)):
        if path is not None and not path.is_file():
            raise DataError(f"file for --{label} not found: {path}")
    for name in ("btc_start", "btc_end", "eth_start", "eth_end"):
        try:
            pd.Timestamp(getattr(args, name))
        except ValueError as exc:
            raise DataError(f"invalid date for --{name.replace('_', '-')}: {getattr(args, name)}") from exc


def parse_time_column(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def load_returns(path, price_column, time_column, start, end):
    try:
        header = pd.read_csv(path, nrows=0).columns
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc

    if price_column not in header:
        raise DataError(f"column '{price_column}' not found in {path}; available: {list(header)}")

    use_time = time_column in header
    columns = [price_column] + ([time_column] if use_time else [])
    df = pd.read_csv(path, usecols=columns)

    if use_time:
        df[time_column] = parse_time_column(df[time_column])
        df = df.dropna(subset=[time_column])
        mask = (df[time_column] >= pd.Timestamp(start)) & (df[time_column] <= pd.Timestamp(end) + pd.Timedelta(days=1))
        df = df.loc[mask].sort_values(time_column)

    prices = pd.to_numeric(df[price_column], errors="coerce")
    prices = prices[prices > 0].dropna()

    if len(prices) < 30:
        raise DataError(f"only {len(prices)} valid prices remain in {path}; at least 30 are required")

    returns = prices.pct_change().to_numpy()[1:]
    returns = returns[np.isfinite(returns)]

    if returns.size < 30:
        raise DataError(f"only {returns.size} valid returns computed from {path}")

    return returns


def fit_distributions(returns, alpha):
    rng = np.random.default_rng(SEED)
    fit_data = returns
    if returns.size > FIT_SAMPLE_SIZE:
        fit_data = rng.choice(returns, FIT_SAMPLE_SIZE, replace=False)

    rows = []
    fitted = {}
    for name, dist in DISTRIBUTIONS.items():
        try:
            params = dist.fit(fit_data)
            ks_stat, ks_p = stats.kstest(returns, name, args=params)
            log_lik = float(np.sum(dist.logpdf(fit_data, *params)))
            aic = 2 * len(params) - 2 * log_lik
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        if not np.isfinite(ks_stat) or not np.isfinite(aic):
            continue
        fitted[name] = params
        rows.append(
            {
                "distribution": name,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_p,
                "aic": aic,
                "reject_at_alpha": ks_p < alpha,
                "parameters": tuple(round(float(p), 8) for p in params),
            }
        )

    if not rows:
        raise DataError("no distribution could be fitted to the data")

    table = pd.DataFrame(rows).sort_values("ks_statistic").reset_index(drop=True)
    table.index += 1
    table.index.name = "rank"
    return table, fitted


def plot_bitcoin_fit(returns, table, fitted, output_dir):
    lower, upper = np.percentile(returns, [0.5, 99.5])
    grid = np.linspace(lower, upper, 600)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(returns, bins=200, range=(lower, upper), density=True, alpha=0.4, color="steelblue", label="Observed")
    for name in table["distribution"].head(3):
        axes[0].plot(grid, DISTRIBUTIONS[name].pdf(grid, *fitted[name]), linewidth=1.6, label=name)
    axes[0].set_title("BTC/USD returns: histogram and fitted PDFs (top 3)")
    axes[0].set_xlabel("Return")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].bar(table["distribution"], table["ks_statistic"], color="steelblue")
    axes[1].set_title("Kolmogorov-Smirnov statistic (lower is better)")
    axes[1].set_ylabel("D")

    fig.tight_layout()
    fig.savefig(output_dir / "btc_distribution_fit.png", dpi=150)
    plt.close(fig)


def analyze_bitcoin(args):
    print("--- Bitcoin distribution fitting ---")
    returns = load_returns(args.btc, args.price_column, args.time_column, args.btc_start, args.btc_end)
    print(f"Returns analysed: {returns.size:,}")
    print(f"Mean: {returns.mean():.3e}  Std: {returns.std(ddof=1):.3e}")
    print(f"Skewness: {stats.skew(returns):.4f}  Excess kurtosis: {stats.kurtosis(returns):.4f}")

    table, fitted = fit_distributions(returns, args.alpha)
    print(table.drop(columns="parameters").to_string(float_format=lambda v: f"{v:.6g}"))
    best = table.iloc[0]["distribution"]
    print(f"Best-fitting distribution (lowest KS statistic): {best}")
    print(f"Best by AIC: {table.sort_values('aic').iloc[0]['distribution']}")

    table.to_csv(args.output_dir / "btc_fit_results.csv")
    plot_bitcoin_fit(returns, table, fitted, args.output_dir)


def plot_ethereum_normality(returns, sample, output_dir):
    lower, upper = np.percentile(returns, [0.5, 99.5])
    grid = np.linspace(lower, upper, 600)
    mu, sigma = returns.mean(), returns.std(ddof=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(returns, bins=200, range=(lower, upper), density=True, alpha=0.4, color="darkorange", label="Observed")
    axes[0].plot(grid, stats.norm.pdf(grid, mu, sigma), color="black", linewidth=1.6, label="Normal")
    axes[0].set_title("ETH/USD returns vs normal PDF")
    axes[0].set_xlabel("Return")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    stats.probplot(sample, dist="norm", plot=axes[1])
    axes[1].set_title(f"Normal Q-Q plot (sample of {sample.size:,})")

    fig.tight_layout()
    fig.savefig(output_dir / "eth_normality.png", dpi=150)
    plt.close(fig)


def analyze_ethereum(args):
    print("\n--- Ethereum normality testing ---")
    returns = load_returns(args.eth, args.price_column, args.time_column, args.eth_start, args.eth_end)
    print(f"Returns analysed: {returns.size:,}")
    print(f"Skewness: {stats.skew(returns):.4f}  Excess kurtosis: {stats.kurtosis(returns):.4f}")

    rng = np.random.default_rng(SEED)
    sample = returns
    if returns.size > SHAPIRO_SAMPLE_SIZE:
        sample = rng.choice(returns, SHAPIRO_SAMPLE_SIZE, replace=False)
        print(f"Shapiro-Wilk is reliable up to {SHAPIRO_SAMPLE_SIZE:,} observations; using a random sample of {sample.size:,}")

    w_stat, w_p = stats.shapiro(sample)
    dag_stat, dag_p = stats.normaltest(returns)
    jb_stat, jb_p = stats.jarque_bera(returns)

    results = pd.DataFrame(
        [
            {"test": "Shapiro-Wilk (sample)", "statistic": w_stat, "p_value": w_p, "n": sample.size},
            {"test": "D'Agostino-Pearson (full)", "statistic": dag_stat, "p_value": dag_p, "n": returns.size},
            {"test": "Jarque-Bera (full)", "statistic": jb_stat, "p_value": jb_p, "n": returns.size},
        ]
    )
    results["decision"] = np.where(results["p_value"] > args.alpha, "fail to reject normality", "reject normality")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    if w_p > args.alpha:
        print("Shapiro-Wilk: fail to reject H0; the returns are consistent with a normal distribution.")
    else:
        print("Shapiro-Wilk: reject H0; the returns are not normally distributed.")

    results.to_csv(args.output_dir / "eth_normality_results.csv", index=False)
    plot_ethereum_normality(returns, sample, args.output_dir)


def main():
    args = parse_args()
    try:
        validate_args(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.btc is not None:
            analyze_bitcoin(args)
        if args.eth is not None:
            analyze_ethereum(args)
    except DataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (OSError, pd.errors.ParserError) as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())