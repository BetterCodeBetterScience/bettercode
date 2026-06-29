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
# ### EVC example
#
# an example of IDE autocompletion

# %%
# implement the expected value of control (EVC) model of Shenhav et al. (2013)

import numpy as np
from scipy.special import softmax
def evc_model(difficulty, reward, cost_weight):
    # compute the expected value of control for each level of control
    control_levels = np.arange(0, 1.1, 0.1)  # control levels from 0 to 1 in steps of 0.1
    evc = reward * (1 - difficulty * (1 - control_levels)) - cost_weight * control_levels**2
    return evc

# example usage
difficulty = 0.5  # task difficulty
reward = 100  # reward for successful performance
cost_weight = 10  # weight of control cost
evc_values = evc_model(difficulty, reward, cost_weight)
print("Control Levels:", np.arange(0, 1.1, 0.1))
print("Expected Value of Control:", evc_values)
# compute the optimal control level
optimal_control_level = np.argmax(evc_values) * 0.1  # multiply by 0.1 to get the actual control level
print("Optimal Control Level:", optimal_control_level)


# %% [markdown]
# API example

# %%
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI"))
system_msg = 'You are a helpful assistant who understands Python programming.'

user_msg = 'Write a python module to implement the expected value of control model by Shenhav et al.' 

chat_completion = client.chat.completions.create(
    messages=[{"role": "user","content": user_msg}],
    model="gpt-5.2",
)
chat_completion.to_dict()['choices'][0]['message']['content'].split("```")

# %%
print(chat_completion.to_dict()['choices'][0]['message']['content'].split("```")[1])

