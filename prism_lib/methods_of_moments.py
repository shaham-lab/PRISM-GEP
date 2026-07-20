import argparse
import os

import numpy as np
from scipy.special import psi, polygamma


def dirichlet_moments(P, shared_precision=True):
    """
    Estimate Dirichlet parameters using method of moments.

    Parameters:
        P: (N, V) matrix, each row a multinomial (sums to 1)

    Returns:
        alpha: estimated Dirichlet parameter vector
    """
    N, V = P.shape

    P = np.maximum(P, 1e-8)

    E_X = np.mean(P, axis=0)
    V_X = np.var(P, axis=0)
    V_X = np.maximum(V_X, 1e-8)

    # Per-dimension concentration estimates s_i = E_i(1-E_i)/V_i - 1.
    per_dim = np.maximum(E_X * (1.0 - E_X) / V_X - 1.0, 1e-7)

    if shared_precision:
        # CORRECT Dirichlet method-of-moments: a SINGLE concentration s shared across
        # all dimensions, so alpha_i = E_i * s is proportional to the mean multinomial
        # E (marker magnitude is preserved). Median collapses the per-dim estimates
        # robustly (ignores the floored / degenerate dims).
        s = np.median(per_dim)
        alpha = E_X * s
    else:
        # LEGACY per-gene precision (BUG): alpha_i = E_i * s_i. Since V_i ~ relvar_i*E_i^2,
        # s_i ~ 1/(relvar_i*E_i), so the E_i marker-magnitude factor CANCELS and
        # alpha_i ~ 1/relvar_i -- a marker-blind prior. Kept only to reproduce pre-fix runs.
        alpha = E_X * per_dim

    alpha = np.maximum(alpha, 0.001)

    return alpha


def dirichlet_minka_fixed_point(P, tol=1e-6, max_iter=1000):
    """
    Estimate Dirichlet parameters using Minka's fixed-point iteration.

    Parameters:
        P: (N, V) matrix, each row a multinomial (sums to 1)

    Returns:
        alpha: estimated Dirichlet parameter vector
    """
    N, V = P.shape
    P = np.maximum(P, 1e-12)

    log_p = np.log(P)
    mean_log_p = np.mean(log_p, axis=0)

    alpha = np.ones(V)

    for _ in range(max_iter):
        alpha0 = np.sum(alpha)
        grad = N * (psi(alpha0) - psi(alpha) + mean_log_p)
        hess = -N * polygamma(1, alpha)
        z = N * polygamma(1, alpha0)
        c = np.sum(grad / hess) / (1.0 / z + np.sum(1.0 / hess))
        update = (grad - c) / hess
        alpha_new = alpha - update
        alpha_new = np.maximum(alpha_new, 1e-6)

        if np.all(np.abs(alpha_new - alpha) < tol):
            alpha = alpha_new
            break

        alpha = alpha_new

    return alpha


def scale_alpha(alpha, a=0.01, b=1.0):
    """
    Scale alpha to be in range [a, b] using min-max scaling.
    """
    alpha_min = alpha.min()
    alpha_max = alpha.max()

    if alpha_max == alpha_min:
        return np.full_like(alpha, fill_value=a)

    return a + (alpha - alpha_min) * (b - a) / (alpha_max - alpha_min)


def load_samples(dataset, metric, n_comp):
    import torch

    path = f"priors/{dataset}/{n_comp}/{metric}_prior.pt"
    samples = torch.load(path)

    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()

    samples = np.asarray(samples).T
    samples = np.maximum(samples, 1e-12)

    row_sums = samples.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    samples = samples / row_sums

    return samples


def generate_synthetic_samples(seed=42, n_samples=1000):
    np.random.seed(seed)
    true_alpha = np.array([2.0, 5.0, 1.0, 0.5, 3.0, 0.01, 0.2, 0.07, 0.01, 0.01])
    samples = np.random.dirichlet(true_alpha, size=n_samples)
    return samples, true_alpha


def main(
    n_comp,
    dataset=None,
    metric=None,
    use_minka=False,
    synthetic=False,
    n_samples=1000,
    seed=42,
):
    if synthetic:
        samples, true_alpha = generate_synthetic_samples(seed=seed, n_samples=n_samples)
        print("True alpha:", true_alpha)

        out_dir = "priors/synthetic"
        out_name = "minka_synthetic.csv" if use_minka else "mom_synthetic.csv"
    else:
        if dataset is None or metric is None:
            raise ValueError("dataset and metric must be provided when synthetic=False")

        samples = load_samples(dataset=dataset, metric=metric, n_comp=n_comp)

        out_dir = f"priors/{dataset}/{n_comp}"
        out_name = f"minka_{metric}.csv" if use_minka else f"mom_{metric}.csv"

    if use_minka:
        estimated_alpha = dirichlet_minka_fixed_point(samples)
        print("Estimated alpha (Minka fixed-point):", estimated_alpha)
    else:
        estimated_alpha = dirichlet_moments(samples)
        print("Estimated alpha (method of moments):", estimated_alpha)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    np.savetxt(out_path, [estimated_alpha], delimiter=",", fmt="%.6f")
    print(f"Saved estimated alpha to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate Dirichlet alpha parameters from prior matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python methods_of_moments.py --n_comp 20 --dataset 20NewsGroup --metric prism
  python methods_of_moments.py --n_comp 5 --dataset BBC --metric pmi --use-minka
  python methods_of_moments.py --n_comp 20 --synthetic
        """,
    )

    parser.add_argument(
        "--n_comp",
        type=int,
        required=True,
        help="Number of components/topics used in the prior path",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name (required unless --synthetic is used)",
    )
    parser.add_argument(
        "--metric",
        help="Metric name used in the saved prior filename (required unless --synthetic is used)",
    )
    parser.add_argument(
        "--use-minka",
        action="store_true",
        help="Use Minka fixed-point estimation instead of method of moments",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic Dirichlet samples instead of loading a saved prior",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data (default: 42)",
    )

    args = parser.parse_args()

    if not args.synthetic:
        if args.dataset is None or args.metric is None:
            parser.error("--dataset and --metric are required unless --synthetic is used")

    main(
        n_comp=args.n_comp,
        dataset=args.dataset,
        metric=args.metric,
        use_minka=args.use_minka,
        synthetic=args.synthetic,
        n_samples=args.n_samples,
        seed=args.seed,
    )


