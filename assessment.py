import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("output_part1")
N_SEARCHES = 10_000
TOLERANCE = 1e-6
MAX_VALUES = 10


def read_number(prompt, low=None, high=None, allow_percent=False):
    while True:
        text = input(prompt).strip()
        try:
            if allow_percent and text.endswith("%"):
                value = float(text[:-1]) / 100
            else:
                value = float(text)
        except ValueError:
            print("Invalid input. Enter a number" + (" (a percentage such as 25% is accepted)." if allow_percent else "."))
            continue
        if not np.isfinite(value):
            print("Value must be finite.")
            continue
        if low is not None and value < low - 1e-12:
            print(f"Value must be at least {low}.")
            continue
        if high is not None and value > high + 1e-12:
            print(f"Value must be at most {high}.")
            continue
        return value


def read_integer(prompt, low, high):
    while True:
        text = input(prompt).strip()
        try:
            value = int(text)
        except ValueError:
            print("Invalid input. Enter a whole number.")
            continue
        if not low <= value <= high:
            print(f"Value must be between {low} and {high}.")
            continue
        return value


def read_vector(prompt, size, low=None, high=None):
    while True:
        parts = input(prompt).replace(",", " ").split()
        if len(parts) != size:
            print(f"Enter exactly {size} values separated by spaces.")
            continue
        try:
            vector = np.array([float(part) for part in parts])
        except ValueError:
            print("Invalid input. All entries must be numbers.")
            continue
        if not np.all(np.isfinite(vector)):
            print("All values must be finite.")
            continue
        if low is not None and np.any(vector < low - 1e-12):
            print(f"All values must be at least {low}.")
            continue
        if high is not None and np.any(vector > high + 1e-12):
            print(f"All values must be at most {high}.")
            continue
        return vector


def read_rates(label, low, high, total):
    while True:
        rates = [
            read_number(f"Enter {label} for Factory {i + 1} ({low} to {high}, or {low * 100:g}% to {high * 100:g}%): ", low, high, True)
            for i in range(3)
        ]
        if abs(sum(rates) - total) <= TOLERANCE:
            return rates
        print(f"Error: the three {label}s must sum to {total}; yours sum to {sum(rates):.6f}. Please re-enter all three.")


def finish_figure(fig, filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.show()
    plt.close(fig)


def describe(values, probs):
    mean = float(np.sum(values * probs))
    variance = float(np.sum((values - mean) ** 2 * probs))
    return mean, variance


def plot_family(values, prob_list, labels, title, filename):
    count = len(prob_list)
    fig, axes = plt.subplots(2, count, figsize=(max(5, 4.5 * count), 7), squeeze=False)
    width = 0.6 * np.min(np.diff(values)) if len(values) > 1 else 0.6
    for col, (probs, label) in enumerate(zip(prob_list, labels)):
        axes[0, col].bar(values, probs, width=width, color="steelblue")
        axes[0, col].set_title(f"{label}: PMF")
        axes[0, col].set_ylim(0, 1.05)
        axes[0, col].set_ylabel("P")
        axes[1, col].step(values, np.cumsum(probs), where="post", color="darkorange")
        axes[1, col].scatter(values, np.cumsum(probs), color="darkorange", zorder=3)
        axes[1, col].set_title(f"{label}: CDF")
        axes[1, col].set_ylim(0, 1.05)
        axes[1, col].set_xlabel("Value")
        axes[1, col].set_ylabel("F(x)")
    fig.suptitle(title)
    finish_figure(fig, filename)


def report(values, probs, label):
    mean, variance = describe(values, probs)
    print(f"{label}: mean = {mean:.4f}, variance = {variance:.4f}")


def item_1_factory_defects():
    print("--- Item 1: Factory Defective Rates ---")
    x = read_rates("production rate", 0.10, 0.40, 1.0)
    y = read_rates("defective rate", 0.01, 0.05, 0.12)
    contributions = [xi * yi for xi, yi in zip(x, y)]
    for i, value in enumerate(contributions, start=1):
        print(f"Factory {i}: x = {x[i - 1]:.4f}, y = {y[i - 1]:.4f}, x*y = {value:.6f}")
    print(f"Probability that a randomly selected product is defective: {sum(contributions):.6f}\n")


def item_2_univariate():
    print("--- Item 2a: Univariate Discrete Distribution ---")
    n = read_integer(f"Number of values (1 to {MAX_VALUES}): ", 1, MAX_VALUES)
    while True:
        values = read_vector(f"Enter {n} distinct values separated by spaces: ", n)
        if len(set(values.tolist())) == n:
            break
        print("Values must be distinct.")
    while True:
        probs = read_vector(f"Enter {n} probabilities separated by spaces: ", n, 0, 1)
        if abs(probs.sum() - 1) <= TOLERANCE:
            break
        print(f"Probabilities must sum to 1; yours sum to {probs.sum():.6f}.")
    order = np.argsort(values)
    values, probs = values[order], probs[order]
    report(values, probs, "Distribution")
    plot_family(values, [probs], ["X"], "Univariate discrete distribution", "univariate.png")
    print()


def read_joint_table(nx, ny):
    while True:
        rows = [read_vector(f"Joint probabilities for X row {i + 1} ({ny} values): ", ny, 0, 1) for i in range(nx)]
        table = np.array(rows)
        if abs(table.sum() - 1) <= TOLERANCE:
            return table
        print(f"The whole table must sum to 1; yours sums to {table.sum():.6f}. Please re-enter all rows.")


def read_distinct_values(label, n):
    while True:
        values = read_vector(f"Enter the {n} distinct values of {label} separated by spaces: ", n)
        if len(set(values.tolist())) == n:
            return np.sort(values)
        print("Values must be distinct.")


def item_2_bivariate():
    print("--- Item 2b: Bivariate Discrete Distribution ---")
    nx = read_integer(f"Number of X values (1 to {MAX_VALUES}): ", 1, MAX_VALUES)
    ny = read_integer(f"Number of Y values (1 to {MAX_VALUES}): ", 1, MAX_VALUES)
    xs = read_distinct_values("X", nx)
    ys = read_distinct_values("Y", ny)
    print("Enter the joint probabilities in ascending order of X (rows) and Y (columns).")
    joint = read_joint_table(nx, ny)

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    print("\nMarginal distributions")
    report(xs, px, "X")
    report(ys, py, "Y")
    plot_family(xs, [px], ["Marginal X"], "Marginal distribution of X", "marginal_x.png")
    plot_family(ys, [py], ["Marginal Y"], "Marginal distribution of Y", "marginal_y.png")

    print("\nConditional distributions of X given Y")
    cond_x, labels_x = [], []
    for j, y_value in enumerate(ys):
        if py[j] <= 0:
            print(f"X | Y={y_value:g}: undefined because P(Y={y_value:g}) = 0")
            continue
        dist = joint[:, j] / py[j]
        cond_x.append(dist)
        labels_x.append(f"X | Y={y_value:g}")
        report(xs, dist, labels_x[-1])
    if cond_x:
        plot_family(xs, cond_x, labels_x, "Conditional distributions of X given Y", "conditional_x_given_y.png")

    print("\nConditional distributions of Y given X")
    cond_y, labels_y = [], []
    for i, x_value in enumerate(xs):
        if px[i] <= 0:
            print(f"Y | X={x_value:g}: undefined because P(X={x_value:g}) = 0")
            continue
        dist = joint[i, :] / px[i]
        cond_y.append(dist)
        labels_y.append(f"Y | X={x_value:g}")
        report(ys, dist, labels_y[-1])
    if cond_y:
        plot_family(ys, cond_y, labels_y, "Conditional distributions of Y given X", "conditional_y_given_x.png")

    independent = np.allclose(joint, np.outer(px, py), atol=1e-9)
    print(f"\nX and Y independent: {'yes' if independent else 'no'}\n")


def pmf_from_samples(samples, support):
    counts = np.array([(samples == k).sum() for k in support])
    return counts / samples.size


def item_3_simulation():
    print("--- Item 3: Search Simulation ---")
    p = read_number("Probability that a site contains the key phrase, 0 < p <= 1 (e.g. 0.6): ", 1e-6, 1)
    rng = np.random.default_rng()
    sims = rng.geometric(p, N_SEARCHES)

    support = np.arange(1, min(int(sims.max()), 30) + 1)
    pmf = pmf_from_samples(sims, support)
    theory = p * (1 - p) ** (support - 1)

    mean, var = sims.mean(), sims.var(ddof=1)
    print(f"Simulated: mean = {mean:.4f}, variance = {var:.4f}")
    print(f"Theoretical: mean = {1 / p:.4f}, variance = {(1 - p) / p ** 2:.4f}")

    cond = sims[sims > 3] - 3
    print(f"\nSearches that failed three times: {cond.size} of {N_SEARCHES}")
    if cond.size < 2:
        print("Too few simulations exceeded 3 searches to study the conditional distribution; try a smaller p.")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(support, pmf, color="steelblue", label="Simulated")
        ax.plot(support, theory, "ko-", label="Theoretical")
        ax.set_title("Simulated PMF of searches to first success")
        ax.set_xlabel("Searches")
        ax.set_ylabel("P")
        ax.legend()
        finish_figure(fig, "simulation_pmf.png")
        return

    cond_pmf = pmf_from_samples(cond, support)
    cond_mean, cond_var = cond.mean(), cond.var(ddof=1)
    print(f"Conditional (X - 3 | X > 3): mean = {cond_mean:.4f}, variance = {cond_var:.4f}")
    print(f"Max |conditional PMF - full PMF| over k = 1..{support[-1]}: {np.max(np.abs(cond_pmf - pmf)):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(support, pmf, color="steelblue", label="Simulated")
    axes[0].plot(support, theory, "ko-", label="Theoretical")
    axes[0].set_title("Simulated PMF of searches to first success")
    axes[0].set_xlabel("Searches")
    axes[0].set_ylabel("P")
    axes[0].legend()
    shift = 0.2
    axes[1].bar(support - shift, pmf, width=2 * shift, color="steelblue", label="Complete set")
    axes[1].bar(support + shift, cond_pmf, width=2 * shift, color="darkorange", label="Given X > 3 (shifted by 3)")
    axes[1].set_title("Conditional vs complete distribution")
    axes[1].set_xlabel("Searches")
    axes[1].set_ylabel("P")
    axes[1].legend()
    finish_figure(fig, "simulation_pmf.png")

    p_gt3 = (sims > 3).mean()
    print("\nMemoryless property checks (simulated vs theoretical):")
    for k_cond, k_plain in ((4, 1), (5, 2)):
        conditional = (sims == k_cond).mean() / p_gt3
        plain = (sims == k_plain).mean()
        exact = p * (1 - p) ** (k_plain - 1)
        print(f"P(X={k_cond} | X>3) = {conditional:.4f}   P(X={k_plain}) = {plain:.4f}   exact P(X={k_plain}) = {exact:.4f}")
    print()


def main():
    try:
        item_1_factory_defects()
        item_2_univariate()
        item_2_bivariate()
        item_3_simulation()
    except (KeyboardInterrupt, EOFError):
        print("\nInput cancelled.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())