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
# data from https://physionet.org/content/bigp3bci/1.0.0/bigP3BCI-data/StudyA/A_01/SE001/Test/CB/#files-panel

# %%
import mne
from pathlib import Path


basedir = Path('/Users/poldrack/data_unsynced/bigp3bci/bigp3bci')

# %%
datafile = basedir / 'Train/CB/A_01_SE001_CB_Train01.edf'

raw = mne.io.read_raw_edf(datafile, preload=True)

# %%
raw

# %%
raw.annotations

# %%
