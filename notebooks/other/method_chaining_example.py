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
# ### Example of method chaining
#
# for workflow chapter

# %%
import pandas as pd

# %%
df_demographics = pd.read_csv('https://raw.githubusercontent.com/IanEisenberg/Self_Regulation_Ontology/refs/heads/master/Data/Complete_02-16-2019/demographics.csv', index_col=0)
df_measures = pd.read_csv('https://raw.githubusercontent.com/IanEisenberg/Self_Regulation_Ontology/refs/heads/master/Data/Complete_02-16-2019/meaningful_variables_clean.csv', index_col=0)

df = df_demographics.join(df_measures)
list(df.columns)

# %%
arrest_stats_by_sex = (df
    .dropna(subset=['Sex', 'ArrestedChargedLifeCount'])
    .replace({'Sex': {0: 'Male', 1: 'Female'}})
    .assign(EverArrested=lambda x: (x['ArrestedChargedLifeCount'] > 0).astype(int))
    .groupby('Sex')
    ['EverArrested']
    .mean()
)

print(arrest_stats_by_sex)

# %%
