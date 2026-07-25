"""Tests for the non-convex optimization landscape example.

The figure these back marks minima on a contour plot, so the invariant that
matters is consistency: the gradient used for descent must be the gradient of
the surface that gets plotted, and any point marked as a minimum must actually
be one.
"""

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from bettercode import optimization_landscape as ol

STARTING_POINTS = [
    (-2.5, 2.5),
    (2.5, -2.5),
    (-0.5, 0.5),
    (1.8, 0.8),
    (-1.8, -0.8),
]

# Centres of the three Gaussian terms in the loss. They are added with positive
# coefficients, so they are barriers between basins, not minima.
BUMP_CENTRES = [(2.0, 1.0), (-1.5, -1.0), (0.5, -2.0)]


def ring_fraction_below(point, radius=0.05, n_samples=72):
    """Fraction of a circle of the given radius around point that has lower loss."""
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    ring = np.array([ol.non_convex_loss(np.asarray(point) + o) for o in offsets])
    return float(np.mean(ring < ol.non_convex_loss(point)))


@pytest.fixture
def grid():
    """The coarse grid the notebook plots the landscape on."""
    return ol.loss_grid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))


@pytest.mark.parametrize("point", [(0.82, -1.38), (-2.75, -2.9), (1.88, 2.48),
                                   (0.64, 1.38), (0.26, 2.61), (0.0, 0.0)])
def test_loss_gradient_matches_numerical_gradient(point):
    """The gradient must describe the surface that is plotted, not an approximation."""
    analytic = ol.loss_gradient(np.array(point))
    numerical = approx_fprime(np.array(point), ol.non_convex_loss, 1e-6)
    assert analytic == pytest.approx(numerical, abs=1e-4)


def test_loss_is_finite_across_the_plotted_domain(grid):
    """No NaNs or infinities anywhere on the plotted surface."""
    _, _, Z = grid
    assert np.isfinite(Z).all()


def test_loss_grid_matches_pointwise_evaluation(grid):
    """The vectorized grid agrees with evaluating the loss point by point."""
    X, Y, Z = grid
    for i, j in [(0, 0), (17, 43), (60, 60), (119, 119)]:
        assert Z[i, j] == pytest.approx(ol.non_convex_loss([X[i, j], Y[i, j]]))


@pytest.mark.parametrize("start", STARTING_POINTS)
def test_gradient_descent_ends_in_a_local_minimum(start):
    """Every trajectory endpoint must sit in a minimum of the plotted surface."""
    path = ol.gradient_descent(np.array(start))
    end = path[-1]
    assert np.linalg.norm(ol.loss_gradient(end)) < 1e-3
    assert ring_fraction_below(end) == 0.0


@pytest.mark.parametrize("start", STARTING_POINTS)
def test_gradient_descent_converges_before_the_iteration_cap(start):
    """Descent settles rather than bouncing until max_iterations."""
    path = ol.gradient_descent(np.array(start), max_iterations=5000)
    assert len(path) < 5000


def test_gradient_descent_path_starts_at_the_starting_point():
    """The returned path is the trajectory, first entry included."""
    start = np.array([-0.5, 0.5])
    path = ol.gradient_descent(start)
    assert path[0] == pytest.approx(start)
    assert path.shape[1] == 2


def test_gradient_descent_finds_distinct_minima():
    """Different starting points get trapped in different minima."""
    ends = [tuple(np.round(ol.gradient_descent(np.array(s))[-1], 2))
            for s in STARTING_POINTS]
    assert len(set(ends)) == len(STARTING_POINTS)


def test_global_minimum_is_a_local_minimum(grid):
    """The point marked with a star must actually be a minimum."""
    X, Y, Z = grid
    point, loss = ol.find_global_minimum(X, Y, Z)
    assert loss == pytest.approx(ol.non_convex_loss(point))
    assert ring_fraction_below(point) == 0.0
    assert np.linalg.norm(ol.loss_gradient(point)) < 1e-3


def test_global_minimum_is_no_worse_than_the_grid_minimum(grid):
    """Polishing the grid argmin may only lower the loss."""
    X, Y, Z = grid
    _, loss = ol.find_global_minimum(X, Y, Z)
    assert loss <= Z.min()


def test_global_minimum_beats_every_descent_endpoint(grid):
    """Trajectories get stuck above the global minimum -- the point of the figure."""
    X, Y, Z = grid
    _, global_loss = ol.find_global_minimum(X, Y, Z)
    for start in STARTING_POINTS:
        end = ol.gradient_descent(np.array(start))[-1]
        assert ol.non_convex_loss(end) > global_loss


@pytest.mark.parametrize("n", [1, 2, 5, 6, 8, 12])
@pytest.mark.parametrize("grayscale", [False, True])
def test_trajectory_styles_returns_one_style_per_trajectory(n, grayscale):
    """However many trajectories there are, every one gets a style."""
    styles = ol.trajectory_styles(n, grayscale=grayscale)
    assert len(styles) == n
    for style in styles:
        assert set(style) == {"color", "linestyle"}


@pytest.mark.parametrize("n", [1, 2, 5, 6, 8, 12])
@pytest.mark.parametrize("grayscale", [False, True])
def test_trajectory_styles_are_mutually_distinguishable(n, grayscale):
    """No two trajectories share both colour and linestyle."""
    styles = ol.trajectory_styles(n, grayscale=grayscale)
    keys = [(s["color"], s["linestyle"]) for s in styles]
    assert len(set(keys)) == n


@pytest.mark.parametrize("n", [1, 5, 6])
def test_trajectory_styles_grayscale_colours_are_achromatic(n):
    """The print variant may not smuggle in colour."""
    from matplotlib.colors import to_rgb

    for style in ol.trajectory_styles(n, grayscale=True):
        r, g, b = to_rgb(style["color"])
        assert r == pytest.approx(g) == pytest.approx(b)


def test_trajectory_styles_vary_linestyle_first_in_grayscale():
    """Linestyle is the channel that survives print, so it separates first."""
    styles = ol.trajectory_styles(5, grayscale=True)
    assert len({s["linestyle"] for s in styles}) == 5


def test_trajectory_styles_vary_colour_first_in_colour():
    """With colour available, keep the lines solid and let hue separate them."""
    styles = ol.trajectory_styles(5)
    assert {s["linestyle"] for s in styles} == {"-"}
    assert len({s["color"] for s in styles}) == 5


def test_trajectory_styles_rejects_nonpositive_counts():
    """A zero or negative count is a caller bug, not an empty plot."""
    with pytest.raises(ValueError):
        ol.trajectory_styles(0)


@pytest.mark.parametrize("centre", BUMP_CENTRES)
def test_gaussian_bump_centres_are_not_minima(centre):
    """Guard against marking the bump centres as minima again.

    Each Gaussian term is added, not subtracted, so its centre is raised above
    its surroundings; descending from it moves away and lowers the loss.
    """
    assert ring_fraction_below(centre, radius=0.2) > 0.0
    end = ol.gradient_descent(np.array(centre))[-1]
    assert ol.non_convex_loss(end) < ol.non_convex_loss(centre)
    assert not np.allclose(end, centre, atol=0.05)
