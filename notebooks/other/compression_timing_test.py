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
# Compare timing of compressed vs uncompressed h5ad files for reading and writing

# %%
import anndata as ad


# %%
# get original data
# %time d_orig = ad.read_h5ad('/Users/poldrack/data_unsynced/BCBS/immune_aging/workflow/checkpoints/step02_filtered.h5ad')

# %%
# save uncompressed version to get timing
# %time d_orig.write_h5ad('/Users/poldrack/data_unsynced/BCBS/immune_aging/workflow/checkpoints/step02_filtered_uncompressed.h5ad', compression=None)

# %% [markdown]
#

# %%
# save compressed version to get timing
# %time d_orig.write_h5ad('/Users/poldrack/data_unsynced/BCBS/immune_aging/workflow/checkpoints/step02_filtered_compressed.h5ad', compression='gzip')

# %%
# load compressed version to get timing
# %time d_comp = ad.read_h5ad('/Users/poldrack/data_unsynced/BCBS/immune_aging/workflow/checkpoints/step02_filtered_compressed.h5ad')

# %%
# !du -sh /Users/poldrack/data_unsynced/BCBS/immune_aging/workflow/checkpoints/step02_filtered*.h5ad

# %%
