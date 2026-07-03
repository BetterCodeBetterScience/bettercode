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


def run_gpt(input, nruns=20):
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
            model="gpt-5.2",
        )
        content.append(chat_completion.to_dict()['choices'][0]['message']['content'].split("```"))
    return content

content_plain = run_gpt(user_msg, 20)

# %%
content_context = []

nruns = 20

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
        model="gpt-5.2",
    )
    content = chat_completion.to_dict()['choices'][0]['message']['content'].split("```")
    content_context.append(content)


# %%
for i, resp in enumerate(content_plain):
    print(f"Run {i+1}: ", [x for x in resp[1].split('\n') if 'def' in x][0])



# %%
for i, resp in enumerate(content_context):
    print(f"Run {i+1}: ", [x for x in resp[1].split('\n') if 'def' in x])


# %%

# %%
def extract_full_function_signature(code_text):
    """Extract the full function signature from code, handling multi-line signatures."""
    lines = code_text.split('\n')
    signature_lines = []
    capturing = False
    paren_count = 0
    
    for line in lines:
        if not capturing and 'def ' in line:
            capturing = True
        
        if capturing:
            signature_lines.append(line.strip())
            # Count opening and closing parentheses
            paren_count += line.count('(') - line.count(')')
            
            # Check if we've closed all parentheses
            if paren_count == 0 and '(' in ' '.join(signature_lines):
                # Signature complete
                break
    
    # Join the lines and clean up extra spaces
    full_signature = ' '.join(signature_lines)
    
    # Remove everything after the first colon that appears after closing )
    if ')' in full_signature:
        close_paren_idx = full_signature.rfind(')')
        # Find the colon after the closing paren
        remaining = full_signature[close_paren_idx:]
        if ':' in remaining:
            colon_idx = close_paren_idx + remaining.index(':')
            full_signature = full_signature[:colon_idx].strip()
    
    return full_signature

# Extract and print full function signatures
print("Full function signatures from content_context:\n")
for i, resp in enumerate(content_context):
    full_sig = extract_full_function_signature(resp[1])
    print(f"Run {i+1}: {full_sig}")
    print()

# %%
# Debug: let's see what the lines look like
print("Lines containing function definition from Run 1:")
lines = content_context[0][1].split('\n')
for i, line in enumerate(lines):
    if 'def ' in line or i > 0 and any('def ' in lines[j] for j in range(max(0, i-5), i)):
        print(f"Line {i}: '{line}'")

# %%
print("Full function signatures from content_plain:\n")
for i, resp in enumerate(content_plain):
    full_sig = extract_full_function_signature(resp[1])
    print(f"Run {i+1}: {full_sig}")
    print()


# %%
