# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: BetterCodeBetterScience
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Mixing languages in a Jupyter notebook
#
# Here we will show an example of mixing Python and R code within a notebook, using *R magic* commands.  We will load and transform a dataset using python, and then perform a multidimensional item response theory (MIRT) analysis using the `mirt` R package.

# %%
import pandas as pd
# %load_ext rpy2.ipython

# %% [markdown]
# Load the individual item response data for the BIS-11 impulsivity survey from Eisenberg et al., 2019

# %%
data_url = 'https://github.com/IanEisenberg/Self_Regulation_Ontology/raw/refs/heads/master/Data/Complete_02-16-2019/Individual_Measures/bis11_survey.csv.gz'
#data_url = '/Users/poldrack/Dropbox/code/Self_Regulation_Ontology/Data/Complete_02-16-2019/Individual_Measures/bis11_survey.csv.gz'
data_df = pd.read_csv(data_url, index_col=0)
item_df = data_df[['question_num', 'text']]
data_df = data_df[['worker_id', 'question_num', 'response']]


# %% [markdown]
# Create a dictionary mapping item numbers to text, for later use.

# %%
# remove duplicates
item_df = item_df.drop_duplicates(subset=['question_num'])

# turn the data frame into a dict
item_num_to_text = {
    f'item_{int(item_df.loc[idx, 'question_num'])}': item_df.loc[idx, 'text']
    for idx in item_df.index
}


# %% [markdown]
# Reformat the data with each item in a separate column

# %%
# rename the question num from "<num>" to "item_<num>" for clarity

data_df['question_num'] = 'item_' + data_df['question_num'].astype(str)

# spread the data with each item in a separate column
# as required by mirt()

data_df_spread = data_df.pivot_table(index='worker_id', 
    columns='question_num', values='response').reset_index(drop=True).dropna()
data_df_spread.shape

# %% [markdown]
# ### R analysis
#
# First, install the require R package if necessary

# %% language="R"
#
# # Perform a multidimensional item response theory (MIRT) analysis using the `mirt` R package
#
# if (!require(mirt)) {
#     install.packages("mirt")
# }
# library(mirt)

# %% [markdown]
# Now pass the data into R using the `-i` flag and perform the MIRT analysis.

# %% magic_args="-i data_df_spread -o bic_values" language="R"
#
# # Perform a multidimensional item response theory (MIRT) analysis using the `mirt` R package
# # Test models with increasing # factors to find the best-fitting model based on minimum BIC
# # This will take a few minutes to run
#
# bic_values <- c()
# n = 1
# best_model_found = FALSE
# fit = list()
#
# while (!best_model_found) {
#     fit[[n]] <- mirt(data_df_spread, n, itemtype = 'graded', SE = TRUE, 
#         verbose = FALSE, method = 'MHRM')
#
#     bic <- extract.mirt(fit[[n]], 'BIC')
#     if (n > 1 && bic > bic_values[length(bic_values)]) {
#         best_model_found = TRUE
#         best_model <- fit[[n - 1]]
#         cat('Best model has', n - 1, 'factor(s) with BIC =', bic_values[length(bic_values)], '\n')
#     } else {
#         cat('Model with', n, 'factor(s): BIC =', bic, '\n')
#         n <- n + 1
#     }
#     bic_values <- c(bic_values, bic)
# }
#

# %% [markdown]
# Get some information about the factors

# %% magic_args="-o loadings" language="R"
# loadings <- as.data.frame(summary(best_model)$rotF, verbose=FALSE)
#
#

# %%
# print strongest loading topn items (pos and neg) for each factor
# excluding items with loading less than 0.5

topn = 3
threshold = 0.5 

for i in range(loadings.shape[1]):
    factor_name = loadings.columns[i]
    print(factor_name)
    factor_df = loadings[[factor_name]]
    factor_df = factor_df.sort_values(by=factor_name)
    for n in range(1, topn+1):
        if factor_df.loc[factor_df.index[-n], factor_name] > threshold:
            print(factor_df.index[-n], item_num_to_text[factor_df.index[-n]], factor_df.loc[factor_df.index[-n], factor_name])
    print('')

# %%
factor_df.loc['item_17', 'F1']

# %% language="R"
#
# empirical_plot(data_df_spread)
