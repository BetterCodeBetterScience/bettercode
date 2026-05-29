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
# Various timing tests for book

# %%
import numpy as np
import pandas as pd
import timeit

# %%
datasize = 1000000
# Capture timing objects
nruns = 1000
results_df = pd.DataFrame(index=['Numpy', 'Pandas', 'List', 'DictValues', 'Set'], columns=['Integer'])

for datatype in ['Integer', 'String']:
    search_val = datasize + 1

    data = np.random.randint(0, datasize, size=datasize)
    if datatype == 'String':
        data = np.array([str(i) for i in data])
        search_val = f"'{search_val}'" # Wrap in quotes for the timeit string

    assert len(data) == datasize
    # need to search for a value that is not present to get worst-case performance
    results_df.loc['Numpy', datatype] = timeit.timeit(f'{search_val} in data', globals=globals(), number=nruns)

    data_pandas = pd.Series(data)
    assert len(data_pandas) == datasize
    results_df.loc['Pandas', datatype] = timeit.timeit(f'{search_val} in data_pandas.values', globals=globals(), number=nruns)
    data_list = data.tolist()
    assert len(data_list) == datasize
    results_df.loc['List', datatype] = timeit.timeit(f'{search_val} in data_list', globals=globals(), number=nruns)

    # don't assert here since set will remove duplicates
    data_set = set(data.tolist())
    results_df.loc['Set', datatype] = timeit.timeit(f'{search_val} in data_set', globals=globals(), number=nruns)

    data_dict = {i:v for i, v in enumerate(data)}
    assert len(data_dict) == datasize
    results_df.loc['DictValues', datatype] = timeit.timeit(f'{search_val} in data_dict.values()', globals=globals(), number=nruns)

print(results_df / nruns * 1e6)  # Convert to microseconds per run

# %%
len(data)

# %%
print((results_df / nruns * 1e6).to_markdown())


# %%
def dotprod_by_hand(a, b):
    return sum([a[i]*b[i] for i in range(len(a))])

npts = 1000
a = np.random.rand(npts)
b = np.random.rand(npts)

# %timeit dotprod_by_hand(a, b)



# %%
# %timeit np.dot(a, b)


# %%
103 / (609 * 1e-3)

# %%
data
