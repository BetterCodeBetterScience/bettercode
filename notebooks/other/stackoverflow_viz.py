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
# Visualization of stack overflow traffic over time, using data obtained from https://data.stackexchange.com/stackoverflow/query/1882532/questions-per-month?ref=blog.pragmaticengineer.com

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
data = pd.read_csv('../data/stackoverflow/QueryResults.csv')
data.columns=['Date', 'Count']
data.head()

# %%
# Convert Date to datetime for proper plotting
data['Date'] = pd.to_datetime(data['Date'])
# make figure wider
plt.figure(figsize=(8, 5))
sns.lineplot(data=data, x='Date', y='Count')
plt.xticks(rotation=45)
plt.ylabel('Number of questions per month')

# WHO declared COVID-19 a pandemic on March 11, 2020
plt.axvline(x=pd.to_datetime('2020-03-11'), color='red', linestyle='--', label='COVID-19 Pandemic Start')
plt.axvline(x=pd.to_datetime('2022-06-21'), color='blue', linestyle='--', label='Github Copilot Release')
plt.axvline(x=pd.to_datetime('2022-11-30'), color='green', linestyle='--', label='ChatGPT Release')

plt.legend()
plt.tight_layout()
plt.savefig('../../book/book/images/stackoverflow_trend.png')

# %%
