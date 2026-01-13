"""
Simulation-Based Inference for Time Series Data

This module provides functionality for performing simulation-based inference (SBI)
on time series data using neural posterior estimation (NPE) with time-series embeddings.

Based on: https://sbi.readthedocs.io/en/latest/how_to_guide/20_time_series_embedding.html
"""

import torch
import numpy as np
from torch import nn
from typing import Optional, Tuple, Dict, Any, Callable

from sbi.inference import NPE
from sbi.neural_nets import embedding_nets, posterior_nn
from sbi.utils import BoxUniform


# ============================================================================
# Simulator Functions
# ============================================================================


def damped_oscillator_simulator(
    theta: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    noise_std: float = 0.05,
) -> torch.Tensor:
    """
    Simulate a damped harmonic oscillator.

    Parameters:
    -----------
    theta : torch.Tensor, shape (batch_size, 3) or (3,)
        Parameters [omega_0, gamma, amplitude]
        - omega_0: natural frequency (0.5 to 3.0 rad/s)
        - gamma: damping coefficient (0.1 to 1.0)
        - amplitude: initial amplitude (0.5 to 2.0)
    t : torch.Tensor or None
        Time points. If None, use 100 points from 0 to 10.
    noise_std : float
        Standard deviation of Gaussian noise

    Returns:
    --------
    x : torch.Tensor, shape (batch_size, n_timesteps) or (n_timesteps,)
        Simulated time series
    """
    if t is None:
        t = torch.linspace(0, 10, 100)

    # Handle both single parameter set and batches
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    omega_0 = theta[:, 0:1]  # shape (batch_size, 1)
    gamma = theta[:, 1:2]
    amplitude = theta[:, 2:3]

    # Damped frequency
    omega = torch.sqrt(torch.clamp(omega_0**2 - gamma**2, min=0.01))

    # Time series: A * exp(-gamma*t) * cos(omega*t)
    t = t.unsqueeze(0)  # shape (1, n_timesteps)
    x = amplitude * torch.exp(-gamma * t) * torch.cos(omega * t)

    # Add Gaussian noise
    noise = torch.randn_like(x) * noise_std
    x = x + noise

    return x.squeeze(0) if theta.shape[0] == 1 else x


def ricker_model(
    N: float, r: float, sigma: float = 0.3, phi: float = 10
) -> Tuple[float, float]:
    """
    Ricker model with stochasticity.
    Defaults from Wood, 2010, Nature

    Parameters:
    -----------
    N : float
        Current population size.
    r : float
        Intrinsic growth rate.
    sigma : float
        Growth rate variability (normally distributed).
    phi : float
        Measurement error term (Poisson distributed).

    Returns:
    --------
    N_next : float
        Next population size.
    y_next : float
        Observed population (with measurement error).
    """
    N_next = r * N * np.exp(-N + np.random.normal(0, sigma))
    y_next = np.random.poisson(N_next * phi)
    return N_next, y_next


def ricker_simulator(
    params: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    n_time_steps: int = 250,
    starting_n: float = 100,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Simulate Ricker population dynamics model.

    Parameters:
    -----------
    params : torch.Tensor, shape (3,)
        Parameters [r, sigma, phi]
        - r: intrinsic growth rate
        - sigma: growth rate variability
        - phi: measurement error term
    t : torch.Tensor or None
        Time points (not used, for API compatibility).
    n_time_steps : int
        Number of time steps to simulate.
    starting_n : float
        Initial population size.
    device : torch.device or None
        Device to place output tensor on. If None, uses params.device.

    Returns:
    --------
    torch.Tensor, shape (n_time_steps,)
        Simulated time series of observed population
    """
    r = params[0].cpu().numpy()
    sigma = params[1].cpu().numpy()
    phi = params[2].cpu().numpy()

    N = starting_n
    timeseries = []
    for _ in range(n_time_steps):
        N, y = ricker_model(N, r, sigma, phi)
        timeseries.append(y)

    if device is None:
        device = params.device
    return torch.as_tensor(np.array(timeseries), dtype=torch.float32, device=device)


# ============================================================================
# Embedding Network Classes
# ============================================================================


class TransformerEmbedding(nn.Module):
    """Transformer-based embedding for time series with global attention."""

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_dim: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length)
        # Reshape to (batch_size, sequence_length, 1) for transformer
        x = x.unsqueeze(-1)

        # Project to d_model dimensions
        x = self.input_projection(x) * np.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # shape: (batch_size, d_model)

        # Project to output dimension
        x = self.fc(x)  # shape: (batch_size, output_dim)

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ============================================================================
# SBI Training and Inference
# ============================================================================


def generate_training_data(
    prior: BoxUniform,
    simulator: Callable,
    n_simulations: int,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training data for SBI.

    Parameters:
    -----------
    prior : BoxUniform
        Prior distribution over parameters.
    simulator : Callable
        Simulator function that takes parameters and returns time series.
    n_simulations : int
        Number of simulations to generate.
    device : torch.device
        Device to use for computation.
    verbose : bool
        Whether to print progress information.

    Returns:
    --------
    thetas : torch.Tensor, shape (n_simulations, n_params)
        Simulated parameters.
    xs : torch.Tensor, shape (n_simulations, n_timesteps)
        Simulated time series.
    """
    if verbose:
        print(f'Generating {n_simulations} simulations for training...')
        print(f'Using device: {device}')

    thetas = prior.sample((n_simulations,)).to(device)
    # Pass device to simulator if it's ricker_simulator
    if hasattr(simulator, '__name__') and simulator.__name__ == 'ricker_simulator':
        xs = torch.stack([simulator(theta, device=device) for theta in thetas])
    else:
        xs = torch.stack([simulator(theta) for theta in thetas]).to(device)

    if verbose:
        print(f'Training data shape: {xs.shape}')
        print(f'Parameters shape: {thetas.shape}')

    return thetas, xs


def build_embedding_network(
    embedding_type: str, sequence_length: int, device: torch.device, **kwargs
) -> nn.Module:
    """
    Build embedding network for time series.

    Parameters:
    -----------
    embedding_type : str
        Type of embedding: 'cnn' or 'transformer'.
    sequence_length : int
        Length of time series.
    device : torch.device
        Device to use for computation.
    **kwargs : dict
        Additional arguments for embedding network.

    Returns:
    --------
    embedding_net : nn.Module
        Embedding network.
    """
    if embedding_type.lower() == 'cnn':
        embedding_net = embedding_nets.CausalCNNEmbedding(
            input_shape=(sequence_length,),
            num_conv_layers=kwargs.get('num_conv_layers', 4),
            pool_kernel_size=kwargs.get('pool_kernel_size', 8),
            output_dim=kwargs.get('output_dim', 20),
        ).to(device)
    elif embedding_type.lower() == 'transformer':
        embedding_net = TransformerEmbedding(
            input_dim=kwargs.get('input_dim', 1),
            d_model=kwargs.get('d_model', 64),
            nhead=kwargs.get('nhead', 4),
            num_layers=kwargs.get('num_layers', 2),
            dim_feedforward=kwargs.get('dim_feedforward', 128),
            output_dim=kwargs.get('output_dim', 20),
            dropout=kwargs.get('dropout', 0.1),
        ).to(device)
    else:
        raise ValueError(f'Unknown embedding type: {embedding_type}')

    return embedding_net


def train_npe(
    prior: BoxUniform,
    thetas: torch.Tensor,
    xs: torch.Tensor,
    embedding_net: nn.Module,
    device: torch.device,
    training_batch_size: int = 50,
    max_num_epochs: int = 100,
    verbose: bool = False,
) -> Tuple[Any, Any]:
    """
    Train Neural Posterior Estimator.

    Parameters:
    -----------
    prior : BoxUniform
        Prior distribution over parameters.
    thetas : torch.Tensor
        Simulated parameters.
    xs : torch.Tensor
        Simulated time series.
    embedding_net : nn.Module
        Embedding network.
    device : torch.device
        Device to use for computation.
    training_batch_size : int
        Batch size for training.
    max_num_epochs : int
        Maximum number of training epochs.
    verbose : bool
        Whether to print progress information.

    Returns:
    --------
    inference : NPE
        Trained inference object.
    posterior : NFlowsFlow
        Trained posterior distribution.
    """
    # Build density estimator
    density_estimator = posterior_nn(
        model='maf',
        embedding_net=embedding_net,
        z_score_x='none',
        z_score_y='none',
    )

    # Create NPE inference object
    inference = NPE(
        prior=prior, density_estimator=density_estimator, device=device
    )

    if verbose:
        print('Training neural posterior estimator...')
        print(f'Training on device: {device}')

    # Add simulations and train
    inference = inference.append_simulations(thetas, xs)
    posterior = inference.train(
        training_batch_size=training_batch_size, max_num_epochs=max_num_epochs
    )

    if verbose:
        print('Training complete!')

    return inference, posterior


def sample_posterior(
    inference: Any,
    posterior: Any,
    x_obs: torch.Tensor,
    n_samples: int = 5000,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Sample from posterior distribution.

    Parameters:
    -----------
    inference : NPE
        Trained inference object.
    posterior : NFlowsFlow
        Trained posterior distribution.
    x_obs : torch.Tensor
        Observed time series.
    n_samples : int
        Number of samples to draw.
    verbose : bool
        Whether to print progress information.

    Returns:
    --------
    posterior_samples : torch.Tensor, shape (n_samples, n_params)
        Samples from posterior distribution.
    """
    if verbose:
        print(f'Drawing {n_samples} samples from posterior...')

    # Build the posterior conditioned on the observation
    posterior_conditioned = inference.build_posterior(posterior)
    posterior_samples = posterior_conditioned.sample((n_samples,), x=x_obs)

    if verbose:
        print(f'Posterior samples shape: {posterior_samples.shape}')

    return posterior_samples


def compute_credible_intervals(
    samples: torch.Tensor,
    true_params: torch.Tensor,
    param_names: list,
    percentiles: list = [5, 25, 50, 75, 95],
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Compute credible intervals for posterior samples.

    Parameters:
    -----------
    samples : torch.Tensor, shape (n_samples, n_params)
        Posterior samples.
    true_params : torch.Tensor, shape (n_params,)
        True parameter values.
    param_names : list
        Names of parameters.
    percentiles : list
        Percentiles to compute.
    verbose : bool
        Whether to print results.

    Returns:
    --------
    results : dict
        Dictionary containing credible intervals and statistics.
    """
    samples_np = samples.cpu().numpy()
    results = {}

    if verbose:
        print('\nPosterior Credible Intervals:')
        print('=' * 70)

    for i, name in enumerate(param_names):
        quantiles = np.percentile(samples_np[:, i], percentiles)
        true_val = true_params[i].item()

        # Check if true value is in 90% credible interval
        in_ci = quantiles[0] <= true_val <= quantiles[4]

        results[name] = {
            'true_value': true_val,
            'median': quantiles[2],
            'mean': samples_np[:, i].mean(),
            'std': samples_np[:, i].std(),
            'ci_50': (quantiles[1], quantiles[3]),
            'ci_90': (quantiles[0], quantiles[4]),
            'in_90_ci': in_ci,
        }

        if verbose:
            print(f'\n{name}:')
            print(f'  True value: {true_val:.3f}')
            print(f'  Posterior median: {quantiles[2]:.3f}')
            print(
                f'  Posterior mean: {samples_np[:, i].mean():.3f} ± {samples_np[:, i].std():.3f}'
            )
            print(f'  50% CI: [{quantiles[1]:.3f}, {quantiles[3]:.3f}]')
            print(f'  90% CI: [{quantiles[0]:.3f}, {quantiles[4]:.3f}]')
            print(f"  True value in 90% CI: {'✓' if in_ci else '✗'}")

    # Calculate mean absolute error
    mae = np.abs(samples_np.mean(axis=0) - true_params.cpu().numpy()).mean()
    results['mae'] = mae

    if verbose:
        print(f"\n{'='*70}")
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f"{'='*70}")

    return results


# ============================================================================
# Main Wrapper Function
# ============================================================================


def run_sbi_timeseries(
    simulator: Callable,
    prior_bounds: Tuple[torch.Tensor, torch.Tensor],
    true_params: torch.Tensor,
    n_simulations: int = 2000,
    n_posterior_samples: int = 5000,
    embedding_type: str = 'cnn',
    sequence_length: int = 250,
    training_batch_size: int = 50,
    max_num_epochs: int = 100,
    param_names: Optional[list] = None,
    device: Optional[torch.device] = None,
    random_seed: int = 42,
    verbose: bool = False,
    **embedding_kwargs,
) -> Dict[str, Any]:
    """
    Run complete SBI pipeline for time series data.

    Parameters:
    -----------
    simulator : Callable
        Simulator function that takes parameters and returns time series.
    prior_bounds : Tuple[torch.Tensor, torch.Tensor]
        Tuple of (low, high) bounds for uniform prior.
    true_params : torch.Tensor
        True parameter values for generating observation.
    n_simulations : int
        Number of training simulations.
    n_posterior_samples : int
        Number of posterior samples to draw.
    embedding_type : str
        Type of embedding: 'cnn' or 'transformer'.
    sequence_length : int
        Length of time series.
    training_batch_size : int
        Batch size for training.
    max_num_epochs : int
        Maximum number of training epochs.
    param_names : list or None
        Names of parameters. If None, uses ['param_0', 'param_1', ...].
    device : torch.device or None
        Device to use. If None, auto-selects CUDA or CPU.
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress information.
    **embedding_kwargs : dict
        Additional arguments for embedding network.

    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'posterior_samples': Posterior samples
        - 'x_obs': Observed time series
        - 'thetas': Training parameters
        - 'xs': Training time series
        - 'inference': Trained inference object
        - 'posterior': Trained posterior
        - 'credible_intervals': Credible interval statistics
        - 'device': Device used
    """
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f'Using device: {device}')

    # Set parameter names
    if param_names is None:
        param_names = [f'param_{i}' for i in range(len(true_params))]

    # Define prior
    prior_low, prior_high = prior_bounds
    prior = BoxUniform(
        low=prior_low.to(device), high=prior_high.to(device), device=device
    )

    # Generate training data
    thetas, xs = generate_training_data(
        prior=prior,
        simulator=simulator,
        n_simulations=n_simulations,
        device=device,
        verbose=verbose,
    )

    # Generate observed data
    x_obs = simulator(true_params).to(device)
    if verbose:
        print(
            f'\nTrue parameters: {dict(zip(param_names, true_params.tolist()))}'
        )
        print(f'Observation shape: {x_obs.shape}')

    # Build embedding network
    embedding_net = build_embedding_network(
        embedding_type=embedding_type,
        sequence_length=sequence_length,
        device=device,
        **embedding_kwargs,
    )

    # Train NPE
    inference, posterior = train_npe(
        prior=prior,
        thetas=thetas,
        xs=xs,
        embedding_net=embedding_net,
        device=device,
        training_batch_size=training_batch_size,
        max_num_epochs=max_num_epochs,
        verbose=verbose,
    )

    # Sample from posterior
    posterior_samples = sample_posterior(
        inference=inference,
        posterior=posterior,
        x_obs=x_obs,
        n_samples=n_posterior_samples,
        verbose=verbose,
    )

    # Compute credible intervals
    credible_intervals = compute_credible_intervals(
        samples=posterior_samples,
        true_params=true_params,
        param_names=param_names,
        verbose=verbose,
    )

    # Return results
    results = {
        'posterior_samples': posterior_samples,
        'x_obs': x_obs,
        'thetas': thetas,
        'xs': xs,
        'inference': inference,
        'posterior': posterior,
        'credible_intervals': credible_intervals,
        'device': device,
        'true_params': true_params,
        'param_names': param_names,
    }

    return results


# ============================================================================
# Convenience Functions for Common Simulators
# ============================================================================


def run_ricker_sbi(
    true_params: Optional[torch.Tensor] = None,
    n_simulations: int = 2000,
    n_posterior_samples: int = 5000,
    embedding_type: str = 'cnn',
    n_time_steps: int = 250,
    starting_n: float = 100,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run SBI for Ricker population model.

    Parameters:
    -----------
    true_params : torch.Tensor or None
        True parameters [r, sigma, phi]. If None, uses [44, 0.3, 10].
    n_simulations : int
        Number of training simulations.
    n_posterior_samples : int
        Number of posterior samples.
    embedding_type : str
        'cnn' or 'transformer'.
    n_time_steps : int
        Number of time steps to simulate.
    starting_n : float
        Initial population size.
    device : torch.device or None
        Device to use.
    verbose : bool
        Print progress information.
    **kwargs : dict
        Additional arguments for run_sbi_timeseries.

    Returns:
    --------
    results : dict
        Results from SBI pipeline.
    """
    if true_params is None:
        true_params = torch.tensor([44.0, 0.3, 10.0])

    # Create simulator wrapper with fixed parameters
    def simulator(params):
        return ricker_simulator(
            params, n_time_steps=n_time_steps, starting_n=starting_n
        )

    # Define prior bounds
    prior_bounds = (
        torch.tensor([1.0, 0.1, 1.0]),
        torch.tensor([100.0, 5.0, 50.0]),
    )

    param_names = [
        'r (growth rate)',
        'sigma (growth noise)',
        'phi (sampling noise)',
    ]

    return run_sbi_timeseries(
        simulator=simulator,
        prior_bounds=prior_bounds,
        true_params=true_params,
        n_simulations=n_simulations,
        n_posterior_samples=n_posterior_samples,
        embedding_type=embedding_type,
        sequence_length=n_time_steps,
        param_names=param_names,
        device=device,
        verbose=verbose,
        **kwargs,
    )


def run_damped_oscillator_sbi(
    true_params: Optional[torch.Tensor] = None,
    n_simulations: int = 2000,
    n_posterior_samples: int = 5000,
    embedding_type: str = 'cnn',
    n_time_steps: int = 100,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run SBI for damped harmonic oscillator.

    Parameters:
    -----------
    true_params : torch.Tensor or None
        True parameters [omega_0, gamma, amplitude]. If None, uses [1.8, 0.4, 1.3].
    n_simulations : int
        Number of training simulations.
    n_posterior_samples : int
        Number of posterior samples.
    embedding_type : str
        'cnn' or 'transformer'.
    n_time_steps : int
        Number of time steps to simulate.
    device : torch.device or None
        Device to use.
    verbose : bool
        Print progress information.
    **kwargs : dict
        Additional arguments for run_sbi_timeseries.

    Returns:
    --------
    results : dict
        Results from SBI pipeline.
    """
    if true_params is None:
        true_params = torch.tensor([1.8, 0.4, 1.3])

    # Create simulator wrapper
    def simulator(params):
        t = torch.linspace(0, 10, n_time_steps)
        return damped_oscillator_simulator(params, t=t)

    # Define prior bounds
    prior_bounds = (
        torch.tensor([0.5, 0.1, 0.5]),
        torch.tensor([3.0, 1.0, 2.0]),
    )

    param_names = ['ω₀ (frequency)', 'γ (damping)', 'A (amplitude)']

    return run_sbi_timeseries(
        simulator=simulator,
        prior_bounds=prior_bounds,
        true_params=true_params,
        n_simulations=n_simulations,
        n_posterior_samples=n_posterior_samples,
        embedding_type=embedding_type,
        sequence_length=n_time_steps,
        param_names=param_names,
        device=device,
        verbose=verbose,
        **kwargs,
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    # Example: Run SBI for Ricker model
    print('Running SBI for Ricker population model...')
    print('=' * 70)

    results = run_ricker_sbi(
        true_params=torch.tensor([44.0, 0.3, 10.0]),
        n_simulations=2000,
        n_posterior_samples=5000,
        embedding_type='cnn',
        n_time_steps=250,
        max_num_epochs=100,
        verbose=False,
    )

    print('\n' + '=' * 70)
    print('SBI complete! Results summary:')
    print(f"  - Posterior samples shape: {results['posterior_samples'].shape}")
    print(f"  - MAE: {results['credible_intervals']['mae']:.4f}")
    print('=' * 70)
