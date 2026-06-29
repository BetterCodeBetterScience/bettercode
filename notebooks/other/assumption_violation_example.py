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
# Examples of the effects of violating independence or normality assumptions

# %% [markdown]
# ## Violating the Independence Assumption
#
# The independence assumption in regression states that observations should be independent of each other - the residuals should not be correlated. When this assumption is violated (e.g., with clustered data, time series, or hierarchical structures), standard errors are typically underestimated, leading to inflated Type I error rates.
#
# We'll demonstrate this by comparing:
# 1. **Independent data**: Where the assumption holds
# 2. **Clustered data**: Where observations within clusters are correlated
# 3. **Time series data**: Where sequential observations are correlated

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Set random seed for reproducibility
np.random.seed(42)
sns.set_style('whitegrid')
