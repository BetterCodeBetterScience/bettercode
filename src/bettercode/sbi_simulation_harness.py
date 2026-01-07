"""
Simulation Harness for SBI Parameter Recovery Experiments

This module provides functionality to run multiple SBI experiments with randomly
generated parameters to evaluate parameter recovery performance.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import json

from .sbi_timeseries import run_ricker_sbi, run_damped_oscillator_sbi


class SBISimulationHarness:
    """
    Harness for running multiple SBI simulations with random parameter sets.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize simulation harness.
        
        Parameters:
        -----------
        output_dir : str or None
            Directory to save results. If None, creates 'sbi_simulations' in current directory.
        device : torch.device or None
            Device to use. If None, auto-selects CUDA or CPU.
        random_seed : int or None
            Random seed for reproducibility. If None, no seed is set.
        """
        self.output_dir = Path(output_dir) if output_dir else Path("sbi_simulations")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        self.results = []
        
        print(f"Simulation harness initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Device: {self.device}")
        
    def generate_random_ricker_params(
        self,
        r_range: Tuple[float, float] = (1.0, 100.0),
        sigma_range: Tuple[float, float] = (0.1, 5.0),
        phi_range: Tuple[float, float] = (1.0, 50.0)
    ) -> torch.Tensor:
        """
        Generate random parameters for Ricker model.
        
        Parameters:
        -----------
        r_range : Tuple[float, float]
            Range for intrinsic growth rate.
        sigma_range : Tuple[float, float]
            Range for growth rate variability.
        phi_range : Tuple[float, float]
            Range for measurement error term.
            
        Returns:
        --------
        params : torch.Tensor, shape (3,)
            Random parameters [r, sigma, phi].
        """
        r = np.random.uniform(*r_range)
        sigma = np.random.uniform(*sigma_range)
        phi = np.random.uniform(*phi_range)
        
        return torch.tensor([r, sigma, phi], dtype=torch.float32)
    
    def generate_random_oscillator_params(
        self,
        omega_0_range: Tuple[float, float] = (0.5, 3.0),
        gamma_range: Tuple[float, float] = (0.1, 1.0),
        amplitude_range: Tuple[float, float] = (0.5, 2.0)
    ) -> torch.Tensor:
        """
        Generate random parameters for damped oscillator.
        
        Parameters:
        -----------
        omega_0_range : Tuple[float, float]
            Range for natural frequency.
        gamma_range : Tuple[float, float]
            Range for damping coefficient.
        amplitude_range : Tuple[float, float]
            Range for initial amplitude.
            
        Returns:
        --------
        params : torch.Tensor, shape (3,)
            Random parameters [omega_0, gamma, amplitude].
        """
        omega_0 = np.random.uniform(*omega_0_range)
        gamma = np.random.uniform(*gamma_range)
        amplitude = np.random.uniform(*amplitude_range)
        
        return torch.tensor([omega_0, gamma, amplitude], dtype=torch.float32)
    
    def run_single_ricker_simulation(
        self,
        sim_id: int,
        true_params: torch.Tensor,
        n_simulations: int = 2000,
        n_posterior_samples: int = 5000,
        embedding_type: str = 'cnn',
        n_time_steps: int = 250,
        max_num_epochs: int = 100,
        verbose: bool = False,
        **kwargs
    ) -> Dict:
        """
        Run single SBI simulation for Ricker model.
        
        Parameters:
        -----------
        sim_id : int
            Simulation identifier.
        true_params : torch.Tensor
            True parameters [r, sigma, phi].
        n_simulations : int
            Number of training simulations.
        n_posterior_samples : int
            Number of posterior samples.
        embedding_type : str
            'cnn' or 'transformer'.
        n_time_steps : int
            Number of time steps to simulate.
        max_num_epochs : int
            Maximum training epochs.
        verbose : bool
            Print detailed progress.
        **kwargs : dict
            Additional arguments for run_ricker_sbi.
            
        Returns:
        --------
        result : dict
            Dictionary containing true parameters, estimates, and errors.
        """
        if verbose:
            print(f"\nSimulation {sim_id}")
            print(f"  True params: r={true_params[0]:.3f}, sigma={true_params[1]:.3f}, phi={true_params[2]:.3f}")
        
        # Run SBI
        sbi_results = run_ricker_sbi(
            true_params=true_params,
            n_simulations=n_simulations,
            n_posterior_samples=n_posterior_samples,
            embedding_type=embedding_type,
            n_time_steps=n_time_steps,
            max_num_epochs=max_num_epochs,
            device=self.device,
            verbose=verbose,
            **kwargs
        )
        
        # Extract results
        posterior_samples = sbi_results['posterior_samples'].cpu().numpy()
        credible_intervals = sbi_results['credible_intervals']
        
        # Compute estimates
        estimates_mean = posterior_samples.mean(axis=0)
        estimates_median = np.median(posterior_samples, axis=0)
        estimates_std = posterior_samples.std(axis=0)
        
        # Compute errors
        true_params_np = true_params.cpu().numpy()
        error_mean = np.abs(estimates_mean - true_params_np)
        error_median = np.abs(estimates_median - true_params_np)
        mae = credible_intervals['mae']
        
        # Check coverage (true params in 90% CI)
        coverage = {
            'r': credible_intervals['r (growth rate)']['in_90_ci'],
            'sigma': credible_intervals['sigma (growth noise)']['in_90_ci'],
            'phi': credible_intervals['phi (sampling noise)']['in_90_ci']
        }
        
        # Store result
        result = {
            'sim_id': sim_id,
            'true_r': float(true_params[0]),
            'true_sigma': float(true_params[1]),
            'true_phi': float(true_params[2]),
            'est_r_mean': float(estimates_mean[0]),
            'est_sigma_mean': float(estimates_mean[1]),
            'est_phi_mean': float(estimates_mean[2]),
            'est_r_median': float(estimates_median[0]),
            'est_sigma_median': float(estimates_median[1]),
            'est_phi_median': float(estimates_median[2]),
            'est_r_std': float(estimates_std[0]),
            'est_sigma_std': float(estimates_std[1]),
            'est_phi_std': float(estimates_std[2]),
            'error_r_mean': float(error_mean[0]),
            'error_sigma_mean': float(error_mean[1]),
            'error_phi_mean': float(error_mean[2]),
            'error_r_median': float(error_median[0]),
            'error_sigma_median': float(error_median[1]),
            'error_phi_median': float(error_median[2]),
            'mae': float(mae),
            'coverage_r': bool(coverage['r']),
            'coverage_sigma': bool(coverage['sigma']),
            'coverage_phi': bool(coverage['phi']),
            'coverage_all': bool(all(coverage.values())),
            'n_simulations': n_simulations,
            'n_posterior_samples': n_posterior_samples,
            'embedding_type': embedding_type,
            'n_time_steps': n_time_steps,
            'max_num_epochs': max_num_epochs
        }
        
        if verbose:
            print(f"  Estimates (mean): r={estimates_mean[0]:.3f}, sigma={estimates_mean[1]:.3f}, phi={estimates_mean[2]:.3f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Coverage: {coverage}")
        
        return result
    
    def run_ricker_parameter_recovery(
        self,
        n_experiments: int,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        n_simulations: int = 2000,
        n_posterior_samples: int = 5000,
        embedding_type: str = 'cnn',
        n_time_steps: int = 250,
        max_num_epochs: int = 100,
        verbose: bool = True,
        save_results: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Run multiple SBI experiments with random Ricker parameters.
        
        Parameters:
        -----------
        n_experiments : int
            Number of experiments to run.
        param_ranges : dict or None
            Dictionary with 'r', 'sigma', 'phi' keys mapping to (min, max) tuples.
            If None, uses default ranges.
        n_simulations : int
            Number of training simulations per experiment.
        n_posterior_samples : int
            Number of posterior samples per experiment.
        embedding_type : str
            'cnn' or 'transformer'.
        n_time_steps : int
            Number of time steps to simulate.
        max_num_epochs : int
            Maximum training epochs per experiment.
        verbose : bool
            Print progress.
        save_results : bool
            Save results to CSV.
        **kwargs : dict
            Additional arguments for SBI.
            
        Returns:
        --------
        results_df : pd.DataFrame
            DataFrame with all results.
        """
        if param_ranges is None:
            param_ranges = {
                'r': (1.0, 100.0),
                'sigma': (0.1, 5.0),
                'phi': (1.0, 50.0)
            }
        
        print(f"Starting {n_experiments} parameter recovery experiments for Ricker model")
        print(f"Parameter ranges:")
        print(f"  r: {param_ranges['r']}")
        print(f"  sigma: {param_ranges['sigma']}")
        print(f"  phi: {param_ranges['phi']}")
        print(f"Device: {self.device}")
        print("=" * 70)
        
        results = []
        
        for i in range(n_experiments):
            # Generate random parameters
            true_params = torch.tensor([
                np.random.uniform(*param_ranges['r']),
                np.random.uniform(*param_ranges['sigma']),
                np.random.uniform(*param_ranges['phi'])
            ], dtype=torch.float32)
            
            if verbose:
                print(f"\n[{i+1}/{n_experiments}] Running experiment...")
            
            # Run simulation
            try:
                result = self.run_single_ricker_simulation(
                    sim_id=i,
                    true_params=true_params,
                    n_simulations=n_simulations,
                    n_posterior_samples=n_posterior_samples,
                    embedding_type=embedding_type,
                    n_time_steps=n_time_steps,
                    max_num_epochs=max_num_epochs,
                    verbose=False,
                    **kwargs
                )
                results.append(result)
                
                if verbose:
                    print(f"  ✓ Complete - MAE: {result['mae']:.4f}, Coverage: {result['coverage_all']}")
                
            except Exception as e:
                print(f"  ✗ Error in experiment {i}: {str(e)}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        if save_results and len(results) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ricker_recovery_{timestamp}.csv"
            filepath = self.output_dir / filename
            results_df.to_csv(filepath, index=False)
            print(f"\nResults saved to: {filepath}")
        
        # Print summary statistics
        if len(results) > 0:
            self._print_summary_statistics(results_df)
        
        self.results = results
        return results_df
    
    def _print_summary_statistics(self, results_df: pd.DataFrame):
        """Print summary statistics of parameter recovery results."""
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        
        print(f"\nNumber of experiments: {len(results_df)}")
        
        # Overall MAE
        print(f"\nMean Absolute Error (MAE):")
        print(f"  Mean: {results_df['mae'].mean():.4f}")
        print(f"  Std:  {results_df['mae'].std():.4f}")
        print(f"  Min:  {results_df['mae'].min():.4f}")
        print(f"  Max:  {results_df['mae'].max():.4f}")
        
        # Coverage rates
        print(f"\nCoverage (% of true params in 90% CI):")
        print(f"  r:     {results_df['coverage_r'].mean()*100:.1f}%")
        print(f"  sigma: {results_df['coverage_sigma'].mean()*100:.1f}%")
        print(f"  phi:   {results_df['coverage_phi'].mean()*100:.1f}%")
        print(f"  All:   {results_df['coverage_all'].mean()*100:.1f}%")
        
        # Per-parameter errors (using mean estimates)
        print(f"\nPer-parameter Mean Absolute Error:")
        print(f"  r:     {results_df['error_r_mean'].mean():.4f} ± {results_df['error_r_mean'].std():.4f}")
        print(f"  sigma: {results_df['error_sigma_mean'].mean():.4f} ± {results_df['error_sigma_mean'].std():.4f}")
        print(f"  phi:   {results_df['error_phi_mean'].mean():.4f} ± {results_df['error_phi_mean'].std():.4f}")
        
        # Relative errors
        print(f"\nRelative Error (% of true value):")
        for param in ['r', 'sigma', 'phi']:
            rel_error = (results_df[f'error_{param}_mean'] / results_df[f'true_{param}']).mean() * 100
            print(f"  {param}: {rel_error:.1f}%")
        
        print("=" * 70)
    
    def save_metadata(self, metadata: Dict, filename: str = "metadata.json"):
        """
        Save metadata about the simulation run.
        
        Parameters:
        -----------
        metadata : dict
            Dictionary containing metadata.
        filename : str
            Filename for metadata file.
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {filepath}")


def run_ricker_recovery_experiment(
    n_experiments: int = 10,
    output_dir: Optional[str] = None,
    n_simulations: int = 2000,
    n_posterior_samples: int = 5000,
    embedding_type: str = 'cnn',
    n_time_steps: int = 250,
    max_num_epochs: int = 100,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    device: Optional[torch.device] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Convenience function to run Ricker parameter recovery experiments.
    
    Parameters:
    -----------
    n_experiments : int
        Number of experiments to run.
    output_dir : str or None
        Directory to save results.
    n_simulations : int
        Number of training simulations per experiment.
    n_posterior_samples : int
        Number of posterior samples per experiment.
    embedding_type : str
        'cnn' or 'transformer'.
    n_time_steps : int
        Number of time steps to simulate.
    max_num_epochs : int
        Maximum training epochs per experiment.
    param_ranges : dict or None
        Custom parameter ranges.
    device : torch.device or None
        Device to use.
    random_seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress.
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with all results.
    """
    # Create harness
    harness = SBISimulationHarness(
        output_dir=output_dir,
        device=device,
        random_seed=random_seed
    )
    
    # Run experiments
    results_df = harness.run_ricker_parameter_recovery(
        n_experiments=n_experiments,
        param_ranges=param_ranges,
        n_simulations=n_simulations,
        n_posterior_samples=n_posterior_samples,
        embedding_type=embedding_type,
        n_time_steps=n_time_steps,
        max_num_epochs=max_num_epochs,
        verbose=verbose,
        save_results=True
    )
    
    # Save metadata
    metadata = {
        'n_experiments': n_experiments,
        'n_simulations': n_simulations,
        'n_posterior_samples': n_posterior_samples,
        'embedding_type': embedding_type,
        'n_time_steps': n_time_steps,
        'max_num_epochs': max_num_epochs,
        'param_ranges': param_ranges,
        'device': str(device) if device else 'auto',
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat()
    }
    harness.save_metadata(metadata)
    
    return results_df


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Run 5 parameter recovery experiments
    print("Running Ricker parameter recovery experiments...")
    print("=" * 70)
    
    results = run_ricker_recovery_experiment(
        n_experiments=5,
        output_dir="sbi_simulations",
        n_simulations=5000,
        n_posterior_samples=10000,
        embedding_type='cnn',
        n_time_steps=250,
        max_num_epochs=100,  # Reduced for faster testing
        random_seed=42,
        verbose=False
    )
    
    print("\n" + "=" * 70)
    print("Experiments complete!")
    print(f"Results shape: {results.shape}")
    print("\nFirst few results:")
    print(results[['sim_id', 'true_r', 'est_r_mean', 'error_r_mean', 'mae', 'coverage_all']].head())
