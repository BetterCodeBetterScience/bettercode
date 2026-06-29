# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## In-context learning
#
# here we show an example of how in-context learning can change the output of a large language model.

# %%
# Import the os package
import os

# Import the openai package
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI"),
)


# %% [markdown]
#

# %%
# Define the system message
system_msg = 'You are a helpful assistant who understands Python programming.'

# Define the user message
user_msg = 'generate a python function to compute a multiple linear regression solution using linear algebra.' 
content_plain = []


def run_gpt(input, nruns=10):
    print(f"Input:", input)
    content = []
    for i in range(nruns):
        # Create a dataset using GPT
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input
                }
            ],
            model="gpt-4o",
        )
        content.append(chat_completion.to_dict()['choices'][0]['message']['content'].split("```"))
    return content

content_plain = run_gpt(user_msg, 10)

# %%
content_context = []

nruns = 10

content1 = "why are type hints important when creating a python function?"
content2 = "generate a python function to compute a multiple linear regression solution using linear algebra."

for i in range(nruns):
    print(f"Run {i+1}/{nruns}")
    # Create a dataset using GPT

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content1
            },
            {
                "role": "user",
                "content": "generate a python function to compute a multiple linear regression solution using linear algebra.",
            }
        ],
        model="gpt-4",
    )
    content = chat_completion.to_dict()['choices'][0]['message']['content'].split("```")
    content_context.append(content)


# %%
for i, resp in enumerate(content_plain):
    print(f"Run {i+1}: ", [x for x in resp[1].split('\n') if 'def' in x][0])



# %%
for i, resp in enumerate(content_context):
    print(f"Run {i+1}: ", [x for x in resp[1].split('\n') if 'def' in x][0])


# %%
print(content_context[4][1])


# %%
print(content_context[2][1])


# %%
