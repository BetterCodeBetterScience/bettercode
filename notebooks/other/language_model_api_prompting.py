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
# In this notebook we will show a simple example of language model prompting using API calls.

# %%
# Import required libraries
import anthropic
import os

# Set up the API client
# Make sure to set your API key as an environment variable: ANTHROPIC_API_KEY
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC")
)


# %% [markdown]
# ### Simple walk-through

# %%
model = "claude-3-5-haiku-latest"
max_tokens = 1000 
prompt = "What is the capital of France?"

message = client.messages.create(
    model=model,
    max_tokens=max_tokens,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

message

# %%
message.content[0].text

# %% [markdown]
# Have the API return the result as a JSON object, containing only the name of the capital.  First let's create a function to submit prompts to claude, in order to simplify our interactions:

# %%
from bettercode.llm_utils import send_prompt_to_claude

json_prompt = """
What is the capital of France? 

Please return your response as a JSON object with the following structure:
{
    "capital": "city_name",
    "country": "country_name"
}
"""

result = send_prompt_to_claude(json_prompt, client)
result

# %%
import json

result_dict = json.loads(result)
result_dict

# %%
# Run on multiple countries

countries = ["France", "Germany", "Spain", "Italy", "Portugal", "Netherlands", "Belgium", "Sweden", "Norway", "Finland"]

ntokens_loop = 0
for country in countries:
    json_prompt = f"""
    What is the capital of {country}? 

    Please return your response as a JSON object with the following structure:
    {{
        "capital": "city_name",
        "country": "country_name"
    }}
    """


    result, ntokens_prompt = send_prompt_to_claude(json_prompt, client, return_tokens=True)
    ntokens_loop += ntokens_prompt
    result_dict = json.loads(result)
    print(result_dict)


# %%
json_prompt_all = f"""
Here is a list of countries:
{', '.join(countries)}

For each country, please provide the capital city in a 
JSON object with the country name as the key and the 
capital city as the value.  

IMPORTANT: Return only the JSON object without any additional text.
"""
result_all, ntokens_prompt = send_prompt_to_claude(json_prompt_all, client, return_tokens=True)


# %%
result_all

# %%
print(ntokens_prompt)
print(ntokens_loop)
