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
# Code for data_management.md

# %%
import pandas as pd
import numpy as np
import os
import templateflow.api as tf
import nibabel as nib


# %%
df = pd.DataFrame({
    'id': [1, 2, 3],
    'height': [170, 180, 175],
    'weight': [70, 80, 75],
    'blood_pressure': [120, 130, 125]
})
print(df.to_markdown(index=False))

# %%
# convert to long format
long_df = pd.melt(df, id_vars=['id'], var_name='measurement', value_name='value')
print(long_df.to_markdown(index=False))

# %% [markdown]
# ### tidy data

# %%
df = pd.DataFrame({
    "site": ["H1", "H2", "H3"],
    "Stage1": np.random.randint(20, 50, size=3),
    "Stage2": np.random.randint(20, 50, size=3),
    "Stage3": np.random.randint(20, 50, size=3),
    "Stage4": np.random.randint(20, 50, size=3),
})
print(df.to_markdown(index=False))

# %%
# melt the df

df_tidy = pd.melt(df, id_vars=["site"], 
    var_name="Stage", value_name="Frequency")
# make stage an integer
df_tidy.Stage = df_tidy.Stage.str.replace("Stage", "").astype(int)
print(df_tidy.to_markdown(index=False))

# %%
df = pd.DataFrame({
    "site": ["H1", "H2", "H3"],
    "Stg1Lng": np.random.randint(20, 50, size=3),
    "Stg2Lng": np.random.randint(20, 50, size=3),
    "Stg3Lng": np.random.randint(20, 50, size=3),
    "Stg4Lng": np.random.randint(20, 50, size=3),
    "Stg1Prs": np.random.randint(20, 50, size=3),
    "Stg2Prs": np.random.randint(20, 50, size=3),
    "Stg3Prs": np.random.randint(20, 50, size=3),
    "Stg4Prs": np.random.randint(20, 50, size=3),
})
print(df.to_markdown(index=False))

# %%
# tidy this, first by melting
df_tidy = pd.melt(df, id_vars=["site"], 
    var_name="Stage_Cancer", value_name="Freq")
# then split Stage_Cancer into two columns
df_tidy[["Stage", "Cancer"]] = df_tidy.Stage_Cancer.str.extract(r'Stg(\d)(\w{3})')
del df_tidy["Stage_Cancer"]
# make Stage an integer
df_tidy.Stage = df_tidy.Stage.astype(int)
# reorder columns
df_tidy = df_tidy[["site", "Stage", "Cancer", "Freq"]]
print(df_tidy.to_markdown(index=False))

# %% [markdown]
# Variables stored in both rows and columns
#

# %%
# gather to create variables in both rows and columns
df_both = df_tidy.pivot_table(index=["site", "Cancer"], 
    columns="Stage", values="Freq").reset_index()
# rename columns to include Stage
df_both.columns.name = None
df_both = df_both.rename(columns={i: f"Stage{i}" for i in range(1,5)})
print(df_both.to_markdown(index=False))

# %%
# gather to make it tidy
df_both_tidy = pd.melt(df_both, id_vars=["site", "Cancer"], 
    var_name="Stage", value_name="Frequency")
# make Stage an integer
df_both_tidy.Stage = df_both_tidy.Stage.str.replace("Stage", "").astype(int)
print(df_both_tidy.to_markdown(index=False))

# %%
df1 = df_both_tidy.query('site=="H1"')
df2 = df_both_tidy.query('site=="H2"')
df3 = df_both_tidy.query('site=="H3"')

# merge data frames
df_merged = pd.concat([df1, df2, df3], ignore_index=True)
print(df_merged.to_markdown(index=False))

# %% [markdown]
# ### Tabular data file formats

# %%
# convert a brain image into a data frame, indexing by x,y,z coordinates

import nibabel as nib
import os
fsldir = os.getenv("FSLDIR")
img = nib.load(os.path.join(fsldir, "data", "standard", "MNI152_T1_2mm.nii.gz"))
data = img.get_fdata()
# get the coordinates of non-zero voxels
coords = np.array(np.nonzero(data)).T
# get the intensity values at those coordinates
intensities = data[coords[:,0], coords[:,1], coords[:,2]]
# create a data frame
df_brain = pd.DataFrame(coords, columns=["x", "y", "z"])
df_brain["intensity"] = intensities
print(df_brain.head().to_markdown(index=False))
print(df_brain.shape)


# %%
# save this to csv and parquet

df_brain.to_csv('/tmp/brain_tabular.csv')
df_brain.to_parquet('/tmp/brain_tabular.parquet')

# %% [markdown]
#

# %%
# !du -sk /tmp/brain_tabular*

# %%
3804/18576


# %%
import time
# time loading of each format
# load 100 times to get average loading time of each format

nreps = 100
start = time.time()
for _ in range(nreps):
    df_csv = pd.read_csv('/tmp/brain_tabular.csv')
end = time.time()
csv_time = (end - start)/nreps
print(f"CSV load time: {csv_time:.4f} seconds")

start = time.time()
for _ in range(nreps):
    df_parquet = pd.read_parquet('/tmp/brain_tabular.parquet')
end = time.time()
parquet_time = (end - start)/nreps
print(f"Parquet load time: {parquet_time:.4f} seconds")
print(f'ratio {csv_time/parquet_time:.2f}')

# %%
os.environ['TEMPLATEFLOW_HOME'] = '/Users/poldrack/.cache/templateflow'

# %%
### Multidimensional arrays

brain = tf.get('MNI152NLin6Asym', resolution=2, extension='.nii.gz',
    desc='brain', suffix='T1w')
img = nib.load(brain)
data = img.get_fdata()


# %% [markdown]
#

# %%
data.shape

# %%
import matplotlib.pyplot as plt
threshold = 0.0001
data[data < threshold] = 0
print(f'proportion of nonzero voxels: {np.mean(data > 0.0001)}')
#plt.imshow(data[:, :, 80, 5], cmap='gray')
plt.imshow(data[:, :, 40], cmap='gray')

plt.savefig('../../book/book/images/difumo_example.png')

# %%
difumo = tf.get('MNI152NLin6Asym', atlas='DiFuMo', desc='512dimensions', 
     resolution=2, suffix='probseg', extension='.nii.gz')
img = nib.load(difumo)
difumo_data = img.get_fdata()

# %%
np.save('/tmp/difumo.npy', difumo_data)
# !du -sm /tmp/difumo.npy


# %%
# save to hdf5
import h5py
with h5py.File('/tmp/difumo.h5', 'w') as f:
    f.create_dataset('difumo', data=difumo_data, compression='gzip')
# !du -sm /tmp/difumo.h5

# %%
# save to zarr
import zarr
zarr_data = zarr.open('/tmp/difumo.zarr', mode='w', 
    shape=difumo_data.shape, dtype=difumo_data.dtype)
zarr_data[:] = difumo_data
# !du -sm /tmp/difumo.zarr

# %%
#compare loading times for each type
import time
nreps = 10

def load_h5py(filename):
    with h5py.File(filename, 'r') as f:
        return f['difumo'][:]

def load_zarr(filename):
    zarr_data = zarr.open(filename, mode='r')
    return zarr_data[:]

loader_funcs = {
    'npy': np.load,
    'h5': load_h5py,
    'zarr': load_zarr
}

for ext in ['npy', 'h5', 'zarr']:
    start_time = time.time()
    filename = f'/tmp/difumo.{ext}'
    for _ in range(nreps):
        data_loaded = loader_funcs[ext](filename)
    end_time = time.time()
    avg_load_time = (end_time - start_time) / nreps
    print(f"Average loading time for {ext}: {avg_load_time:.6f} seconds")

# %% [markdown]
# ### Graph data

# %%
import networkx as nx

friends = [
    ('Bill', 'Sally'),
    ('Bill', 'Mark'),
    ('Bill', 'Elise'),
    ('Mark', 'Elise'),
    ('Mark', 'Lisa')
]
G = nx.Graph()
G.add_edges_from(friends)
G.edges

# %%
# plot spring-embeddding
import matplotlib.pyplot as plt
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=800, node_color='lightblue', font_size=10)
plt.savefig('../../book/images/graph_example.png')

# %%
# generate adjacency matrix
adj_matrix = nx.adjacency_matrix(G).todense()
labels = list(G.nodes)
adj_df = pd.DataFrame(adj_matrix, index=labels, columns=labels)
print(adj_df.to_markdown())

# %% [markdown]
# ### File granularity

# %%
data = np.random.randn(10000, 100000)
data.shape

# %%
tmpdir = '/tmp/array_granularity/'
if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)

# save as single large file
np.save(os.path.join(tmpdir, 'data_large.npy'), data)

# save as multiple small files
for i in range(data.shape[0]):
    np.save(os.path.join(tmpdir, f'data_row_{i:04d}.npy'), data[i, :])

# test loading time
import time
start = time.time()
data_loaded_large = np.load(os.path.join(tmpdir, 'data_large.npy'))
end = time.time()
large_load_time = end - start
print(f"Loading time for large file: {large_load_time:.4f} seconds")

start = time.time()
data_loaded_small = np.array([np.load(os.path.join(tmpdir, f'data_row_{i:04d}.npy')) for i in range(data.shape[0])])
end = time.time()
small_load_time = end - start
print(f"Loading time for small files: {small_load_time:.4f} seconds")


# %% [markdown]
# ### File naming

# %%
filename = 'sub-001_sess-1A_desc-Diffusion_fa.nii.gz'

def split_filename(filename):
    extension = '.'.join(filename.split('.')[1:])
    name = filename.split('.')[0]
    key_values = {k:v for k,v in (item.split('-') for item in name.split('_')[:-1])}
    key_values['suffix'] = name.split('_')[-1]
    return extension, key_values

extension, key_values = split_filename(filename)
pprint(key_values)




# %% [markdown]
# ### DataLad
#

# %%
# %cd ../..
# %pwd

# !rm -rf my_datalad_repo

# %% vscode={"languageId": "shellscript"}
# script to create a datalad repository, download data files, modify one of them, and save the changes
sudo rm -rf my_datalad_repo # sometimes requires sudo
datalad create my_datalad_repo
# cd my_datalad_repo
# mkdir data

datalad download-url --dataset . -O data/ \
  https://raw.githubusercontent.com/IanEisenberg/Self_Regulation_Ontology/refs/heads/master/Data/Complete_02-16-2019/meaningful_variables_clean.csv 
datalad download-url --dataset . -O data/ \
  https://raw.githubusercontent.com/IanEisenberg/Self_Regulation_Ontology/refs/heads/master/Data/Complete_02-16-2019/demographics.csv

datalad unlock data/demographics.csv

python ../src/bettercode/modify_data.py data/demographics.csv
datalad save -m "removed Motivation variables from demographics.csv"
datalad status



# %%
# remove Motivation variables from demographics.csv
df = pd.read_csv('demographics.csv')
df = df.loc[:, ~df.columns.str.contains('Motivation')]
df.to_csv('demographics.csv', index=False)

# %%
# !datalad status
