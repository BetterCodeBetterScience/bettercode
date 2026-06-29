# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: bettercode
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Effects of Violations of Independence Assumption
#
# This notebook demonstrates how violations of the independence assumption affect statistical testing under the null hypothesis (no signal). We'll compare:
#
# 1. **IID (Independent and Identically Distributed) data**: Properly independent observations
# 2. **Non-IID data**: Data with dependency structure (e.g., autocorrelation)
#
# Both scenarios will have the same noise level, differing only in independence structure.

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.linear_model import GLSAR
from statsmodels.tsa.ar_model import AutoReg
from arch.covariance.kernel import Bartlett

from pathlib import Path

FIGURE_DIR = Path('../../book/book/images')

# Set random seed for reproducibility
np.random.seed(1)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# %% [markdown]
# ## 1. Data Generation Functions
#
# We'll create two types of data:
# - **IID data**: Simple random normal noise
# - **Non-IID data**: Autocorrelated (AR(1)) process with the same marginal variance

# %%
def generate_iid_data(n_samples, noise_std=1.0, seed=None):
    """
    Generate independent and identically distributed data under null hypothesis.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise_std : float
        Standard deviation of the noise
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    data : ndarray
        IID data with no signal (pure noise)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate pure noise (null hypothesis: no signal)
    data = np.random.normal(0, noise_std, n_samples)
    return data


def generate_non_iid_data(n_samples, noise_std=1.0, autocorr=0.7, seed=None):
    """
    Generate non-IID data with autocorrelation (AR(1) process) under null hypothesis.
    The marginal variance is adjusted to match the IID case.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise_std : float
        Target standard deviation (marginal)
    autocorr : float
        Autocorrelation coefficient (0 to 1)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    data : ndarray
        Non-IID autocorrelated data with no signal
    """
    if seed is not None:
        np.random.seed(seed)
    
    # For AR(1): X_t = rho * X_{t-1} + epsilon_t
    # Marginal variance: Var(X) = sigma_epsilon^2 / (1 - rho^2)
    # So to get target variance noise_std^2, we need:
    # sigma_epsilon = noise_std * sqrt(1 - rho^2)
    
    innovation_std = noise_std * np.sqrt(1 - autocorr**2)
    
    # Generate AR(1) process
    data = np.zeros(n_samples)
    data[0] = np.random.normal(0, noise_std)  # Start from stationary distribution
    
    for t in range(1, n_samples):
        innovation = np.random.normal(0, innovation_std)
        data[t] = autocorr * data[t-1] + innovation
    
    return data


# %% [markdown]
# ## 2. Visualize Sample Data
#
# Let's generate and visualize examples of both types of data to see the difference.

# %%
# Generate sample data
n = 500
noise_std = 1.0

iid_sample = generate_iid_data(n, noise_std=noise_std, seed=123)
non_iid_sample = generate_non_iid_data(n, noise_std=noise_std, autocorr=0.7, seed=123)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time series plots
axes[0, 0].plot(iid_sample, alpha=0.7, linewidth=1)
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0, 0].set_title('IID Data (Independent)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(non_iid_sample, alpha=0.7, linewidth=1, color='orange')
axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Non-IID Data (Autocorrelated)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Sample Index')
axes[0, 1].set_ylabel('Value')
axes[0, 1].grid(True, alpha=0.3)

# Histograms
axes[1, 0].hist(iid_sample, bins=30, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[1, 0].set_title(f'IID Distribution (std={np.std(iid_sample):.3f})', fontsize=12)
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(non_iid_sample, bins=30, alpha=0.7, edgecolor='black', color='orange')
axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].set_title(f'Non-IID Distribution (std={np.std(non_iid_sample):.3f})', fontsize=12)
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print(f"IID data - Mean: {np.mean(iid_sample):.4f}, Std: {np.std(iid_sample):.4f}")
print(f"Non-IID data - Mean: {np.mean(non_iid_sample):.4f}, Std: {np.std(non_iid_sample):.4f}")


# %% [markdown]
# ## 3. Autocorrelation Analysis
#
# Let's examine the autocorrelation structure to confirm the difference in independence.

# %%
def compute_autocorrelation(data, max_lag=20):
    """
    Compute autocorrelation function.
    
    Parameters:
    -----------
    data : ndarray
        Time series data
    max_lag : int
        Maximum lag to compute
        
    Returns:
    --------
    acf : ndarray
        Autocorrelation function values
    """
    acf = np.zeros(max_lag + 1)
    n = len(data)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n
    
    for k in range(max_lag + 1):
        ck = np.sum((data[:-k or None] - mean) * (data[k:] - mean)) / n
        acf[k] = ck / c0
    
    return acf


# Compute autocorrelations
max_lag = 20
acf_iid = compute_autocorrelation(iid_sample, max_lag)
acf_non_iid = compute_autocorrelation(non_iid_sample, max_lag)

# Plot autocorrelations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# IID autocorrelation
ax1.bar(range(max_lag + 1), acf_iid, alpha=0.7, edgecolor='black')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
# Add 95% confidence bounds (approximately 1.96/sqrt(n))
confidence_bound = 1.96 / np.sqrt(len(iid_sample))
ax1.axhline(y=confidence_bound, color='r', linestyle='--', alpha=0.5, label='95% CI')
ax1.axhline(y=-confidence_bound, color='r', linestyle='--', alpha=0.5)
ax1.set_title('Autocorrelation: IID Data', fontsize=14, fontweight='bold')
ax1.set_xlabel('Lag')
ax1.set_ylabel('Autocorrelation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Non-IID autocorrelation
ax2.bar(range(max_lag + 1), acf_non_iid, alpha=0.7, edgecolor='black', color='orange')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.axhline(y=confidence_bound, color='r', linestyle='--', alpha=0.5, label='95% CI')
ax2.axhline(y=-confidence_bound, color='r', linestyle='--', alpha=0.5)
ax2.set_title('Autocorrelation: Non-IID Data', fontsize=14, fontweight='bold')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. Statistical Testing Under the Null Hypothesis
#
# Now we'll perform many simulations and apply statistical tests to each dataset. We'll compare six approaches:
# 1. **OLS/t-test on IID data**: Should maintain nominal Type I error rate (5%)
# 2. **OLS/t-test on non-IID data**: Will show inflated Type I error rate
# 3. **GLS with AR(1) correction**: Attempts to correct but may be inadequate
# 4. **Prewhitened HAC with sandwich estimator**: Fit AR to residuals, apply kernel HAC to prewhitened scores, recolor, and use sandwich formula for standard errors
# 5. **Iterative FGLS**: Iteratively estimate AR parameters and transform data until convergence
# 6. **Parametric Bootstrap**: Simulate under H0 with estimated AR structure to get empirical p-values
#
# Under the null hypothesis (no signal), proper methods should maintain α ≈ 0.05.

# %%
def prewhitened_hac_inference(y, X, ar_order=1):
    """
    OLS with prewhitened HAC standard errors.
    
    Steps:
    1. Fit OLS, get residuals
    2. Fit AR(p) to residuals to prewhiten
    3. Apply kernel HAC to prewhitened residuals
    4. Recolor to get long-run covariance
    5. Compute sandwich standard errors
    """
    n, k = X.shape
    
    # Step 1: OLS fit
    ols_result = OLS(y, X).fit()
    resid = ols_result.resid
    
    # Step 2: Fit AR model to residuals (prewhitening)
    ar_model = AutoReg(resid, lags=ar_order, old_names=False).fit()
    ar_coefs = ar_model.params[1:]  # exclude intercept
    
    # Prewhitened residuals: e_t - sum(phi_i * e_{t-i})
    prewhitened = resid[ar_order:].copy()
    for i in range(ar_order):
        prewhitened -= ar_coefs[i] * resid[ar_order - 1 - i : n - 1 - i]
    
    # Step 3: Compute score contributions (X * residuals)
    X_trimmed = X[ar_order:]
    scores = X_trimmed * prewhitened[:, None]
    
    # Step 4: HAC on prewhitened scores
    kernel_est = Bartlett(scores, df_adjust=k)
    S_prewhitened = kernel_est.cov.long_run
    
    # Step 5: Recolor - multiply by (1 - sum(phi))^(-2)
    ar_sum = np.sum(ar_coefs)
    recolor_factor = 1 / (1 - ar_sum) ** 2
    S_recolored = S_prewhitened * recolor_factor
    
    # Step 6: Sandwich formula: (X'X)^{-1} S (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)
    cov_matrix = n * XtX_inv @ S_recolored @ XtX_inv
    
    se = np.sqrt(np.diag(cov_matrix))
    t_stats = ols_result.params / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
    
    return {
        'params': ols_result.params,
        'se': se,
        't_stats': t_stats,
        'p_values': p_values,
        'ar_coefs': ar_coefs,
        'bandwidth': kernel_est.bandwidth,
    }


def fgls_ar(y, X, ar_order=1, max_iter=10, tol=1e-6):
    """
    Iterative FGLS with AR(p) errors.
    """
    n, k = X.shape
    params = OLS(y, X).fit().params
    
    for _ in range(max_iter):
        resid = y - X @ params
        
        # Fit AR to residuals
        ar_model = AutoReg(resid, lags=ar_order, old_names=False).fit()
        phi = ar_model.params[1:]
        
        # Transform data: quasi-difference
        y_trans = y[ar_order:].copy()
        X_trans = X[ar_order:].copy()
        for i in range(ar_order):
            y_trans -= phi[i] * y[ar_order - 1 - i : n - 1 - i]
            X_trans -= phi[i] * X[ar_order - 1 - i : n - 1 - i]
        
        # Re-estimate on transformed data
        new_params = OLS(y_trans, X_trans).fit().params
        
        if np.max(np.abs(new_params - params)) < tol:
            break
        params = new_params
    
    # Final inference on transformed model
    final_fit = OLS(y_trans, X_trans).fit()
    
    return {
        'params': final_fit.params,
        'se': final_fit.bse,
        't_stats': final_fit.tvalues,
        'p_values': final_fit.pvalues,
        'ar_coefs': phi,
    }


def parametric_bootstrap_test(y, X, n_boot=999, seed=None):
    """
    Parametric bootstrap for regression with AR errors.
    Simulates under H0: beta=0 with estimated AR structure.
    """
    rng = np.random.default_rng(seed)
    n, k = X.shape
    
    # Fit model and estimate AR structure
    ols_fit = OLS(y, X).fit()
    resid = ols_fit.resid
    ar_fit = AutoReg(resid, lags=1, old_names=False).fit()
    rho = ar_fit.params[1]
    sigma = np.std(ar_fit.resid)
    
    # Observed t-statistic
    t_obs = ols_fit.tvalues
    
    # Bootstrap under H0
    t_boot = np.zeros((n_boot, k))
    for b in range(n_boot):
        # Generate AR(1) errors
        eps = rng.normal(0, sigma, n)
        errors = np.zeros(n)
        errors[0] = eps[0] / np.sqrt(1 - rho**2)  # stationary start
        for t in range(1, n):
            errors[t] = rho * errors[t-1] + eps[t]
        
        # y = 0 + errors under H0
        y_boot = errors
        
        boot_fit = OLS(y_boot, X).fit()
        t_boot[b] = boot_fit.tvalues
    
    # Two-sided p-values
    p_values = np.mean(np.abs(t_boot) >= np.abs(t_obs), axis=0)
    
    return {
        'params': ols_fit.params,
        't_obs': t_obs,
        'p_values': p_values,
        'rho_hat': rho,
    }


def run_simulation(n_simulations=1000, n_samples=50, noise_std=1.0, 
                   autocorr=0.7, alpha=0.05, seed=None):
    """
    Run multiple simulations to assess Type I error rates.
    Compares OLS, GLS, Prewhitened HAC, iterative FGLS, and parametric bootstrap.
    
    Parameters:
    -----------
    n_simulations : int
        Number of simulation runs
    n_samples : int
        Sample size for each simulation
    noise_std : float
        Standard deviation of noise
    autocorr : float
        Autocorrelation coefficient for non-IID data
    alpha : float
        Significance level
    seed : int, optional
        Random seed
        
    Returns:
    --------
    results : dict
        Dictionary containing p-values and Type I error rates
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Store p-values from all simulations
    p_values_iid = []
    p_values_non_iid_ols = []
    p_values_non_iid_gls = []
    p_values_non_iid_prewhitened = []
    p_values_non_iid_fgls = []
    p_values_non_iid_bootstrap = []
    
    for i in range(n_simulations):
        # Generate data
        iid_data = generate_iid_data(n_samples, noise_std=noise_std)
        non_iid_data = generate_non_iid_data(n_samples, noise_std=noise_std, 
                                             autocorr=autocorr)
        
        # Test 1: OLS on IID data (standard t-test)
        t_stat_iid, p_val_iid = stats.ttest_1samp(iid_data, 0)
        p_values_iid.append(p_val_iid)
        
        # Test 2: OLS on non-IID data (standard t-test - incorrect for dependent data)
        t_stat_non_iid, p_val_non_iid_ols = stats.ttest_1samp(non_iid_data, 0)
        p_values_non_iid_ols.append(p_val_non_iid_ols)
        
        # Test 3: GLS with AR(1) correction on non-IID data (may be inadequate)
        try:
            X = np.ones((n_samples, 1))
            y = non_iid_data
            glsar_model = GLSAR(y, X, rho=1)
            glsar_results = glsar_model.iterative_fit(maxiter=5)
            p_val_non_iid_gls = glsar_results.pvalues[0]
            p_values_non_iid_gls.append(p_val_non_iid_gls)
        except:
            p_values_non_iid_gls.append(np.nan)
        
        # Test 4: Prewhitened HAC with sandwich estimator
        try:
            X = np.ones((n_samples, 1))
            y = non_iid_data
            result = prewhitened_hac_inference(y, X, ar_order=1)
            p_val_non_iid_prewhitened = result['p_values'][0]
            p_values_non_iid_prewhitened.append(p_val_non_iid_prewhitened)
        except:
            p_values_non_iid_prewhitened.append(np.nan)
        
        # Test 5: Iterative FGLS with AR(1)
        try:
            X = np.ones((n_samples, 1))
            y = non_iid_data
            result = fgls_ar(y, X, ar_order=1)
            p_val_non_iid_fgls = result['p_values'][0]
            p_values_non_iid_fgls.append(p_val_non_iid_fgls)
        except:
            p_values_non_iid_fgls.append(np.nan)
        
        # Test 6: Parametric bootstrap
        try:
            X = np.ones((n_samples, 1))
            y = non_iid_data
            result = parametric_bootstrap_test(y, X, n_boot=999)
            p_val_non_iid_bootstrap = result['p_values'][0]
            p_values_non_iid_bootstrap.append(p_val_non_iid_bootstrap)
        except:
            p_values_non_iid_bootstrap.append(np.nan)
    
    # Remove any NaN values
    p_values_non_iid_gls = np.array([p for p in p_values_non_iid_gls if not np.isnan(p)])
    p_values_non_iid_prewhitened = np.array([p for p in p_values_non_iid_prewhitened if not np.isnan(p)])
    p_values_non_iid_fgls = np.array([p for p in p_values_non_iid_fgls if not np.isnan(p)])
    p_values_non_iid_bootstrap = np.array([p for p in p_values_non_iid_bootstrap if not np.isnan(p)])
    
    # Calculate Type I error rates (false positive rate)
    type1_error_iid = np.mean(np.array(p_values_iid) < alpha)
    type1_error_non_iid_ols = np.mean(np.array(p_values_non_iid_ols) < alpha)
    type1_error_non_iid_gls = np.mean(p_values_non_iid_gls < alpha) if len(p_values_non_iid_gls) > 0 else np.nan
    type1_error_non_iid_prewhitened = np.mean(p_values_non_iid_prewhitened < alpha) if len(p_values_non_iid_prewhitened) > 0 else np.nan
    type1_error_non_iid_fgls = np.mean(p_values_non_iid_fgls < alpha) if len(p_values_non_iid_fgls) > 0 else np.nan
    type1_error_non_iid_bootstrap = np.mean(p_values_non_iid_bootstrap < alpha) if len(p_values_non_iid_bootstrap) > 0 else np.nan
    
    results = {
        'p_values_iid': np.array(p_values_iid),
        'p_values_non_iid_ols': np.array(p_values_non_iid_ols),
        'p_values_non_iid_gls': p_values_non_iid_gls,
        'p_values_non_iid_prewhitened': p_values_non_iid_prewhitened,
        'p_values_non_iid_fgls': p_values_non_iid_fgls,
        'p_values_non_iid_bootstrap': p_values_non_iid_bootstrap,
        'type1_error_iid': type1_error_iid,
        'type1_error_non_iid_ols': type1_error_non_iid_ols,
        'type1_error_non_iid_gls': type1_error_non_iid_gls,
        'type1_error_non_iid_prewhitened': type1_error_non_iid_prewhitened,
        'type1_error_non_iid_fgls': type1_error_non_iid_fgls,
        'type1_error_non_iid_bootstrap': type1_error_non_iid_bootstrap
    }
    
    return results


# %% [markdown]
# ## 5. Run Simulations and Analyze Results

# %%
# Run simulations
print("Running simulations...")
results = run_simulation(n_simulations=1000, n_samples=50, noise_std=1.0, 
                        autocorr=0.7, alpha=0.05, seed=42)

print("\n" + "="*80)
print("RESULTS: Type I Error Rates (False Positive Rates)")
print("="*80)
print(f"Expected Type I error rate (α): 5.00%")
print()
print(f"1. OLS on IID data (correct baseline):        {results['type1_error_iid']*100:.2f}%")
print(f"2. OLS on non-IID data (WRONG):               {results['type1_error_non_iid_ols']*100:.2f}%")
print(f"3. GLS (AR1) on non-IID data (inadequate):    {results['type1_error_non_iid_gls']*100:.2f}%")
print(f"4. Prewhitened HAC:                           {results['type1_error_non_iid_prewhitened']*100:.2f}%")
print(f"5. Iterative FGLS:                            {results['type1_error_non_iid_fgls']*100:.2f}%")
print(f"6. Parametric Bootstrap:                      {results['type1_error_non_iid_bootstrap']*100:.2f}%")
print()
print(f"OLS inflation factor (non-IID/IID):        {results['type1_error_non_iid_ols']/results['type1_error_iid']:.2f}x")
print(f"GLS (still inadequate):                    {results['type1_error_non_iid_gls']/results['type1_error_iid']:.2f}x nominal")
print(f"Prewhitened HAC:                           {results['type1_error_non_iid_prewhitened']/results['type1_error_iid']:.2f}x nominal")
print(f"Iterative FGLS:                            {results['type1_error_non_iid_fgls']/results['type1_error_iid']:.2f}x nominal")
print(f"Parametric Bootstrap:                      {results['type1_error_non_iid_bootstrap']/results['type1_error_iid']:.2f}x nominal")
print("="*80)

# %% [markdown]
# ## 6. Visualize P-value Distributions
#
# Note: We show only the three most informative comparisons.

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# P-value histograms
axes[0, 0].hist(results['p_values_iid'], bins=20, alpha=0.7, edgecolor='black', 
                range=(0, 1))
axes[0, 0].axhline(y=len(results['p_values_iid'])/20, color='r', linestyle='--', 
                   alpha=0.5, label='Uniform expectation')
axes[0, 0].axvline(x=0.05, color='orange', linestyle='--', linewidth=2, 
                   label='α = 0.05')
axes[0, 0].set_title('OLS on IID Data\n(Correct)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('P-value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(results['p_values_non_iid_ols'], bins=20, alpha=0.7, edgecolor='black', 
                color='orange', range=(0, 1))
axes[0, 1].axhline(y=len(results['p_values_non_iid_ols'])/20, color='r', linestyle='--', 
                   alpha=0.5, label='Uniform expectation')
axes[0, 1].axvline(x=0.05, color='darkred', linestyle='--', linewidth=2, 
                   label='α = 0.05')
axes[0, 1].set_title('OLS on Non-IID Data\n(WRONG - Inflated)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('P-value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].hist(results['p_values_non_iid_prewhitened'], bins=20, alpha=0.7, edgecolor='black', 
                color='green', range=(0, 1))
axes[0, 2].axhline(y=len(results['p_values_non_iid_prewhitened'])/20, color='r', linestyle='--', 
                   alpha=0.5, label='Uniform expectation')
axes[0, 2].axvline(x=0.05, color='darkgreen', linestyle='--', linewidth=2, 
                   label='α = 0.05')
axes[0, 2].set_title('Prewhitened HAC on Non-IID\n(AR + Bartlett)', fontsize=13, fontweight='bold')
axes[0, 2].set_xlabel('P-value')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend(fontsize=9)
axes[0, 2].grid(True, alpha=0.3)

# Cumulative distribution of p-values
sorted_p_iid = np.sort(results['p_values_iid'])
sorted_p_non_iid_ols = np.sort(results['p_values_non_iid_ols'])
sorted_p_non_iid_prewhitened = np.sort(results['p_values_non_iid_prewhitened'])
cumulative_iid = np.arange(1, len(sorted_p_iid) + 1) / len(sorted_p_iid)
cumulative_ols = np.arange(1, len(sorted_p_non_iid_ols) + 1) / len(sorted_p_non_iid_ols)
cumulative_prewhitened = np.arange(1, len(sorted_p_non_iid_prewhitened) + 1) / len(sorted_p_non_iid_prewhitened)

axes[1, 0].plot(sorted_p_iid, cumulative_iid, linewidth=2, label='OLS on IID')
axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Uniform (expected)')
axes[1, 0].axvline(x=0.05, color='orange', linestyle='--', linewidth=2, alpha=0.7)
axes[1, 0].set_title('Cumulative Distribution:\nOLS on IID', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('P-value')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim([0, 1])
axes[1, 0].set_ylim([0, 1])

axes[1, 1].plot(sorted_p_non_iid_ols, cumulative_ols, linewidth=2, color='orange', 
                label='OLS on Non-IID')
axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Uniform (expected)')
axes[1, 1].axvline(x=0.05, color='darkred', linestyle='--', linewidth=2, alpha=0.7)
axes[1, 1].set_title('Cumulative Distribution:\nOLS on Non-IID', fontsize=12, 
                     fontweight='bold')
axes[1, 1].set_xlabel('P-value')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim([0, 1])
axes[1, 1].set_ylim([0, 1])

axes[1, 2].plot(sorted_p_non_iid_prewhitened, cumulative_prewhitened, linewidth=2, color='green', 
                label='Prewhitened HAC on Non-IID')
axes[1, 2].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Uniform (expected)')
axes[1, 2].axvline(x=0.05, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
axes[1, 2].set_title('Cumulative Distribution:\nPrewhitened HAC', fontsize=12,
                     fontweight='bold')
axes[1, 2].set_xlabel('P-value')
axes[1, 2].set_ylabel('Cumulative Probability')
axes[1, 2].legend(fontsize=9)
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_xlim([0, 1])
axes[1, 2].set_ylim([0, 1])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Effect of Autocorrelation Strength
#
# Let's examine how different levels of autocorrelation affect the Type I error rate.

# %%
# Test different autocorrelation levels
autocorr_levels = np.linspace(0, 0.9, 10)
type1_errors_ols = []
type1_errors_gls = []
type1_errors_prewhitened = []
type1_errors_fgls = []
type1_errors_bootstrap = []

print("Testing different autocorrelation levels...")
for autocorr in autocorr_levels:
    result = run_simulation(n_simulations=500, n_samples=50, noise_std=1.0,
                          autocorr=autocorr, alpha=0.05)
    type1_errors_ols.append(result['type1_error_non_iid_ols'])
    type1_errors_gls.append(result['type1_error_non_iid_gls'])
    type1_errors_prewhitened.append(result['type1_error_non_iid_prewhitened'])
    type1_errors_fgls.append(result['type1_error_non_iid_fgls'])
    type1_errors_bootstrap.append(result['type1_error_non_iid_bootstrap'])
    print(f"Autocorr: {autocorr:.2f} -> OLS: {result['type1_error_non_iid_ols']*100:.2f}%, "
          f"GLS: {result['type1_error_non_iid_gls']*100:.2f}%, "
          f"Prewhitened: {result['type1_error_non_iid_prewhitened']*100:.2f}%, "
          f"FGLS: {result['type1_error_non_iid_fgls']*100:.2f}%, "
          f"Bootstrap: {result['type1_error_non_iid_bootstrap']*100:.2f}%")



# %%
# Plot results
plt.figure(figsize=(12, 7))
plt.plot(autocorr_levels, type1_errors_ols, 'o-', linewidth=2, markersize=8, 
         color='orange', label='OLS')
plt.plot(autocorr_levels, type1_errors_gls, 's-', linewidth=2, markersize=7, 
         color='gray', alpha=0.6, label='GLS with AR(1)')
plt.plot(autocorr_levels, type1_errors_prewhitened, '^-', linewidth=2, markersize=8, 
         color='green', label='Prewhitened HAC')
plt.plot(autocorr_levels, type1_errors_fgls, 'd-', linewidth=2, markersize=7, 
         color='purple', label='Iterative FGLS')
plt.plot(autocorr_levels, type1_errors_bootstrap, 'v-', linewidth=2, markersize=7, 
         color='blue', label='Parametric Bootstrap')
plt.axhline(y=0.05, color='r', linestyle='--', linewidth=2, label='Nominal α = 0.05')
plt.xlabel('Autocorrelation (ρ)', fontsize=16)
plt.ylabel('Type I Error Rate', fontsize=16)
plt.title('Effect of Autocorrelation on Type I Error Rate', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=14, loc='upper left')
plt.grid(True, alpha=0.3)
plt.ylim([0, max(max(type1_errors_ols), max(type1_errors_gls)) * 1.1])

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'type1_error_vs_autocorr.png', dpi=300)
plt.show()


# %% [markdown]
# ## 8. Summary and Interpretation
#
# ### Key Findings:
#
# 1. **OLS on IID Data**: When independence assumptions are met, the Type I error rate stays close to the nominal level (α = 0.05), and p-values follow a uniform distribution under the null hypothesis.
#
# 2. **OLS on Non-IID Data (VERY WRONG)**: When independence is violated (autocorrelated data), standard OLS/t-tests produce severely inflated Type I error rates (up to 68% with high autocorrelation!). This is a common and serious mistake in data analysis.
#
# 3. **GLS with AR(1) Correction (INADEQUATE)**: While GLS attempts to correct for autocorrelation, it often fails to fully control Type I error rates, especially with:
#    - High autocorrelation (still ~27% error rate at ρ=0.9)
#    - Small sample sizes
#    - Misspecified correlation structure
#    - Estimation uncertainty in the autocorrelation parameter
#    
#    **GLS is better than ignoring the problem, but not a complete solution!**
#
# 4. **Prewhitened HAC with Sandwich Estimator**: Uses a sophisticated multi-step approach:
#    - Fits OLS and obtains residuals
#    - Fits AR model to residuals (prewhitening filter)
#    - Applies prewhitening filter to create score contributions (X * prewhitened residuals)
#    - Applies Bartlett kernel HAC to prewhitened scores
#    - Recolors the covariance by multiplying by (1 - sum(AR coefs))^(-2)
#    - Uses sandwich formula: (X'X)^(-1) S (X'X)^(-1) for robust covariance
#
# 5. **Iterative FGLS**: Feasible Generalized Least Squares with iterative refinement:
#    - Starts with OLS to get initial residuals
#    - Fits AR model to residuals
#    - Transforms (quasi-differences) the data using AR parameters
#    - Re-estimates on transformed data
#    - Repeats until convergence
#    - Final inference uses standard errors from the transformed model
#
# ### Why Each Method Performs as It Does:
#
# **OLS (wrong)**:
# - Assumes independence, so standard errors are too small
# - Results in overconfident inference and inflated Type I errors
#
# **GLS (inadequate)**:
# - Requires estimating the autocorrelation parameter from the data
# - Uses the same data for both estimation and inference (double-dipping)
# - Can fail when autocorrelation is high or model is misspecified
# - Small sample performance is poor
#
# **Prewhitened HAC with Sandwich Estimator (robust hybrid approach)**:
# - Combines parametric (AR) and nonparametric (kernel) methods in a principled way
# - Prewhitening removes most autocorrelation, making kernel estimation more effective
# - The Bartlett kernel provides robustness to remaining dependencies in prewhitened scores
# - Recoloring step accounts for the transformation back to original scale
# - Sandwich formula properly accounts for all sources of uncertainty
# - Should maintain proper Type I error control across autocorrelation structures
#
# **Iterative FGLS (parametric transformation approach)**:
# - Iteratively refines AR parameter estimates and transforms data
# - Quasi-differencing removes autocorrelation from the data itself
# - Standard inference is valid on the transformed (whitened) data
# - Converges to consistent estimates when AR model is correctly specified
# - Performance depends on accurate AR model and sufficient iterations
#
# ### Practical Recommendations:
#
# For autocorrelated/time series data:
#
# 1. **Prewhitened HAC with sandwich estimator**
#    - Use the `prewhitened_hac_inference()` function (shown in this notebook)
#    - Combines AR prewhitening with kernel HAC and proper sandwich formula
#    - More efficient than pure kernel methods when autocorrelation structure is well-approximated by AR
#    - Properly accounts for uncertainty in all estimation steps
#
# 2. **Iterative FGLS**
#    - Use the `fgls_ar()` function (shown in this notebook)
#    - Iteratively transforms data to remove autocorrelation
#    - Provides valid inference when AR structure is correctly specified
#    - More efficient when model is correct, but less robust to misspecification
#
# 3. **Alternative approaches:**
#    - **Block bootstrap**: Preserve dependency structure in resampling
#    - **Mixed-effects models**: For hierarchical/grouped data structures
#    - **Time series models (ARIMA/SARIMAX)**: If modeling the time series itself
#    - **Cluster-robust standard errors**: For grouped/panel data
#
# 4. **Don't use:**
#    - Standard OLS/t-tests on autocorrelated data (completely invalid)
#    - GLS alone unless verified via simulation
#
# 5. **Always:**
#    - Plot your data and examine the autocorrelation structure
#    - Report the ACF alongside your results
#    - Consider whether the autocorrelation is substantive or a nuisance
#    - Verify your inference approach via simulation when possible
#
# ### Key Takeaways:
#
# 1. **Standard OLS/t-tests are completely invalid for autocorrelated data** - Type I error rates can exceed 60%!
#
# 2. **Prewhitened HAC sandwich combines three approaches** - AR prewhitening + kernel HAC + sandwich covariance formula
#
# 3. **Iterative FGLS provides an alternative** - Transforms data iteratively to remove autocorrelation
#
# 4. **Trade-offs exist** - FGLS is more efficient when correct but less robust; HAC is more robust but may be less efficient
#
# 5. **Proper implementation matters** - The recoloring step (HAC) and convergence criteria (FGLS) are critical for valid inference
#
# 6. **The challenge is real** - Accounting for autocorrelation while maintaining valid inference requires sophisticated methods
#
# **Bottom line**: When you have autocorrelated data, use methods that explicitly account for the dependency structure. Both prewhitened HAC sandwich and iterative FGLS provide principled approaches, with HAC being more robust and FGLS being more efficient when correctly specified.

# %%
# Calculate effective sample size for autocorrelated data
def effective_sample_size(n, acf, max_lag=None):
    """
    Estimate effective sample size accounting for autocorrelation.
    Uses the formula: n_eff = n / (1 + 2 * sum(acf))
    """
    if max_lag is None:
        max_lag = len(acf) - 1
    
    # Sum autocorrelations (excluding lag 0)
    acf_sum = np.sum(acf[1:max_lag+1])
    n_eff = n / (1 + 2 * acf_sum)
    return n_eff

# Calculate for our example
n_actual = 50
example_non_iid = generate_non_iid_data(n_actual, noise_std=1.0, autocorr=0.7, seed=123)
acf_example = compute_autocorrelation(example_non_iid, max_lag=10)
n_eff = effective_sample_size(n_actual, acf_example)

print(f"\nEffective Sample Size Analysis:")
print(f"Actual sample size: {n_actual}")
print(f"Effective sample size (with ρ=0.7): {n_eff:.1f}")
print(f"Information loss: {(1 - n_eff/n_actual)*100:.1f}%")

# %% [markdown]
# ### Clustered data

# %%
