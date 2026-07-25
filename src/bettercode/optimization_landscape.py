"""A rough non-convex loss landscape, for demonstrating local minima.

Pedagogical example for the validation chapter: a two-parameter loss surface
built from a quadratic bowl, three Gaussian barriers, and a stack of
oscillatory terms, plus plain gradient descent over it. Starting points in
different basins get trapped in different local minima.

The gradient is taken numerically from ``non_convex_loss`` rather than derived
by hand. That keeps a single source of truth for the surface: a hand-derived
gradient that omits or mis-signs a term sends descent somewhere that is not a
minimum of the surface being plotted, which is invisible until the figure looks
wrong.
"""

from __future__ import annotations

import numpy as np

Point = np.ndarray

# Centres, weights, and widths of the three Gaussian terms. These are ADDED to
# the loss, so each is a barrier (a bump) separating basins -- not a minimum.
_BARRIERS = (
    (2.0, 1.0, 2.0, 0.5),
    (-1.5, -1.0, 1.5, 0.3),
    (0.5, -2.0, 1.8, 0.4),
)


def non_convex_loss(params: Point | list[float]) -> np.ndarray | float:
    """Loss at ``params`` = [x, y]; accepts scalars or broadcastable arrays."""
    x, y = params

    barriers = sum(
        weight * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / width)
        for cx, cy, weight, width in _BARRIERS
    )

    oscillations = (
        0.5 * np.sin(6 * x) * np.cos(4 * y)
        + 0.4 * np.sin(8 * y) * np.cos(7 * x)
        + 0.3 * np.sin(10 * x + 5 * y) * np.cos(6 * x - 3 * y)
        + 0.2 * np.sin(12 * x) * np.sin(10 * y)
        + 0.15 * np.cos(15 * x) * np.cos(13 * y)
    )

    # Localized roughness and cross-frequency modulation, so the basins are not
    # all alike.
    local_features = (
        0.3 * np.exp(-((x - 1) ** 2 + (y - 0.5) ** 2) / 0.2) * np.sin(18 * x)
        + 0.25 * np.exp(-((x + 0.8) ** 2 + (y + 1.2) ** 2) / 0.15) * np.cos(20 * y)
        + 0.15 * np.sin(5 * x) * np.sin(3 * y) * np.cos(8 * x * y)
        + 0.1 * np.cos(6 * x + 2 * y) * np.sin(9 * x - 4 * y)
    )

    return 0.5 * (x**2 + y**2) + barriers + oscillations + local_features


def loss_gradient(params: Point, eps: float = 1e-6) -> Point:
    """Central-difference gradient of ``non_convex_loss`` at ``params``."""
    x, y = np.asarray(params, dtype=float)
    dx = (non_convex_loss([x + eps, y]) - non_convex_loss([x - eps, y])) / (2 * eps)
    dy = (non_convex_loss([x, y + eps]) - non_convex_loss([x, y - eps])) / (2 * eps)
    return np.array([dx, dy])


def loss_grid(
    x_range: np.ndarray, y_range: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the loss over a meshgrid, returning (X, Y, Z)."""
    X, Y = np.meshgrid(x_range, y_range)
    return X, Y, non_convex_loss([X, Y])


def gradient_descent(
    start_point: Point,
    learning_rate: float = 0.01,
    max_iterations: int = 5000,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Run fixed-step gradient descent, returning the path including the start.

    The step size has to stay small enough for the oscillatory terms: at
    ``learning_rate=0.03`` some starting points overshoot the fine structure
    and keep bouncing until ``max_iterations`` instead of settling in a
    minimum.
    """
    current = np.asarray(start_point, dtype=float).copy()
    path = [current.copy()]

    for _ in range(max_iterations):
        gradient = loss_gradient(current)
        if np.linalg.norm(gradient) < tolerance:
            break
        current = current - learning_rate * gradient
        path.append(current.copy())

    return np.array(path)


def trajectory_styles(
    n_trajectories: int, grayscale: bool = False
) -> list[dict[str, str]]:
    """Colour and linestyle for each trajectory, for any number of them.

    Every trajectory gets a distinct (colour, linestyle) pair, so callers can
    add starting points without silently losing curves. The channel that varies
    first is the one that carries the most information for the variant: hue in
    colour (lines stay solid), linestyle in grayscale. The second channel only
    starts cycling once the first is exhausted.
    """
    from bettercode.figstyle import LINESTYLES, gray_palette

    if n_trajectories < 1:
        raise ValueError(f"n_trajectories must be at least 1, got {n_trajectories}")

    styles = list(LINESTYLES)
    if grayscale:
        colors = gray_palette(4)  # beyond four shades the grays stop separating
    else:
        colors = _colorblind_hex(10)

    out = []
    for i in range(n_trajectories):
        if grayscale:
            style_idx, color_idx = i % len(styles), (i // len(styles)) % len(colors)
        else:
            color_idx, style_idx = i % len(colors), (i // len(colors)) % len(styles)
        out.append({"color": colors[color_idx], "linestyle": styles[style_idx]})
    return out


def _colorblind_hex(n: int) -> list[str]:
    """n colours from seaborn's colorblind palette, as hex strings."""
    import seaborn as sns

    return [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b in sns.color_palette("colorblind", n)
    ]


def find_global_minimum(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> tuple[Point, float]:
    """Locate the lowest point of the landscape: grid search, then polished.

    The grid argmin is only accurate to the grid spacing, which is enough to
    pick the right basin but leaves the marker visibly off the minimum; a local
    optimization from there settles it.
    """
    from scipy.optimize import minimize

    idx = np.unravel_index(np.argmin(Z), Z.shape)
    seed = np.array([X[idx], Y[idx]])
    result = minimize(non_convex_loss, seed, method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12})
    if result.fun > Z[idx]:  # polishing should never make it worse
        return seed, float(Z[idx])
    return result.x, float(result.fun)
