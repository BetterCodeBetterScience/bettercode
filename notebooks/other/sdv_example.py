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
# example of using SDV to generate synthetic data based on the demographic and sruvey data from Eisenberg et al.

# %%
import sdv
import pandas as pd

from pathlib import Path

FIGURE_DIR = Path("../../book/book/images")

# %%
df_demographics = pd.read_csv('https://raw.githubusercontent.com/IanEisenberg/Self_Regulation_Ontology/refs/heads/master/Data/Complete_02-16-2019/demographics.csv', index_col=0).replace({'Sex': {0: 'Male', 1: 'Female'}})
variables_of_interest = ['Sex', 'Age', 'HighestEducation', 
       'HeightInches', 'WeightPounds','ArrestedChargedLifeCount',
       'CoffeeCupsPerDay', 'HouseholdIncome']
df_demographics = df_demographics[variables_of_interest]

df_measures = pd.read_csv('https://raw.githubusercontent.com/IanEisenberg/Self_Regulation_Ontology/refs/heads/master/Data/Complete_02-16-2019/meaningful_variables_clean.csv', index_col=0)
measures_of_interest = [
       'upps_impulsivity_survey.lack_of_perseverance',
       'upps_impulsivity_survey.lack_of_premeditation',
       'upps_impulsivity_survey.negative_urgency',
       'upps_impulsivity_survey.positive_urgency',
       'upps_impulsivity_survey.sensation_seeking',
        'ten_item_personality_survey.agreeableness',
       'ten_item_personality_survey.conscientiousness.ReflogTr',
       'ten_item_personality_survey.emotional_stability',
       'ten_item_personality_survey.extraversion',
       'ten_item_personality_survey.openness',
       'ravens.score',
       'cognitive_reflection_survey.correct_proportion'
]
df_measures = df_measures[measures_of_interest]
df_orig = df_demographics.join(df_measures, how='inner')


df_orig.head() 

# %%
# Create synthetic data with independent variables using SDV
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import numpy as np
import pandas as pd
import warnings

def generate_independent_synthetic_data(df, random_seed=42):
    """
    Generate synthetic data where all variables are independent.
    
    Uses SDV to model the full dataset, then shuffles each column 
    independently to break all correlations while preserving marginal distributions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe to generate synthetic version of
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    pd.DataFrame
        Synthetic dataframe with same shape and column names as input,
        but with independent variables
    """
    # Suppress the metadata saving warning
    warnings.filterwarnings('ignore', message='We strongly recommend saving the metadata')
    
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create metadata for the full dataset
    metadata = Metadata.detect_from_dataframe(
        data=df,
        table_name='full_data'
    )
    
    # Create synthesizer for the full dataset
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_rounding=False,
        enforce_min_max_values=True,
        default_distribution='norm'
    )
    
    # Fit synthesizer to the full dataset
    synthesizer.fit(df)
    
    # Generate synthetic data
    df_synthetic = synthesizer.sample(num_rows=len(df))
    
    # CRITICAL: Shuffle each column independently to break all correlations
    # This preserves the marginal distribution of each variable but eliminates dependencies
    for col in df_synthetic.columns:
        shuffled_values = df_synthetic[col].values.copy()
        np.random.shuffle(shuffled_values)
        df_synthetic[col] = shuffled_values
    
    return df_synthetic


# Generate synthetic data
print("Generating synthetic data with independent variables using SDV...")
df_synthetic = generate_independent_synthetic_data(df_orig, random_seed=42)

print(f"\nOriginal data shape: {df_orig.shape}")
print(f"Synthetic data shape: {df_synthetic.shape}")
print("\nFirst few rows of synthetic data:")
df_synthetic.head()

# %%
# Compare distributions and correlations between original and synthetic data
import matplotlib.pyplot as plt
import seaborn as sns

# Check that correlations are indeed minimal in synthetic data
numeric_orig = df_orig.select_dtypes(include=[np.number])
numeric_synth = df_synthetic.select_dtypes(include=[np.number])

print("Original data correlations (first 5x5):")
print(numeric_orig.corr().iloc[:5, :5].round(3))
print("\nSynthetic data correlations (first 5x5):")
print(numeric_synth.corr().iloc[:5, :5].round(3))

# plot correlation matrices

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(numeric_orig.corr(), ax=axes[0], cmap='coolwarm', vmin=-1, vmax=1, 
            xticklabels=False, yticklabels=False, cbar_kws={'label': 'Correlation'})
axes[0].set_title('Original Data Correlations')
sns.heatmap(numeric_synth.corr(), ax=axes[1], cmap='coolwarm', vmin=-1, vmax=1,
            xticklabels=False, yticklabels=False, cbar_kws={'label': 'Correlation'})
axes[1].set_title('Synthetic Data Correlations')
plt.tight_layout()
plt.savefig(FIGURE_DIR / "sdv_correlations.png")

# Visualize distributions of a few variables
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

numeric_cols = numeric_orig.columns[:6]

for i, col in enumerate(numeric_cols):
    if col in df_synthetic.columns:
        orig_data = df_orig[col].dropna()
        synth_data = df_synthetic[col].dropna()
        
        axes[i].hist(orig_data, alpha=0.5, label='Original', bins=30, density=True, color='blue')
        axes[i].hist(synth_data, alpha=0.5, label='Synthetic', bins=30, density=True, color='orange')
        axes[i].set_title(col, fontsize=10)
        axes[i].legend()
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')

plt.tight_layout()
plt.savefig(FIGURE_DIR / "sdv_distributions.png")
plt.show()

# Print summary statistics comparison
print("\n" + "="*80)
print("Summary statistics comparison for first few variables:")
print("="*80)
for col in numeric_cols[:3]:
    print(f"\n{col}:")
    print(f"  Original: mean={df_orig[col].mean():.2f}, std={df_orig[col].std():.2f}")
    print(f"  Synthetic: mean={df_synthetic[col].mean():.2f}, std={df_synthetic[col].std():.2f}")

# %%
df_orig[['HouseholdIncome', 'cognitive_reflection_survey.correct_proportion']].corr()

# %%
from scipy.stats import spearmanr


# Compute correlation and p-value for HouseholdIncome and cognitive_reflection_survey.correct_proportion
col1 = 'HouseholdIncome'
col2 = 'cognitive_reflection_survey.correct_proportion'

# Drop NaN values for the correlation calculation
data = df_orig[[col1, col2]].dropna()

# Compute Pearson correlation and p-value
corr, pval = spearmanr(data[col1], data[col2])

print(f"Correlation between {col1} and {col2}:")
print(f"  Spearman r = {corr:.4f}")
print(f"  p-value = {pval:.5f}")
print(f"  n = {len(data)}")

# %%
# perform randomization analysis to confirm significance

n_permutations = 10000
observed_corr, _ = spearmanr(data[col1], data[col2])
# include observed corr to ensure that p is not zero
permuted_corrs = [observed_corr]
for _ in range(n_permutations):
    shuffled_col2 = data[col2].sample(frac=1, replace=False).values
    permuted_corr, _ = spearmanr(data[col1], shuffled_col2)
    permuted_corrs.append(permuted_corr)

p_value_randomization = np.mean(np.abs(permuted_corrs) >= np.abs(observed_corr))
print(f"Randomization p-value = {p_value_randomization:.5f}")
