from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    pipeline,
)
from datetime import datetime
import torch
print("Start Time:", datetime.now().strftime("%H:%M:%S"))

model_name = "sagard21/python-code-explainer"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)

model.eval()

pipe = pipeline("summarization", model=model_name, config=config, tokenizer=tokenizer)

raw_code = """
def is_prime(n):
    # Check if n is less than 2
    if n < 2:
        return False
    # Check for factors from 2 up to √n
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True
"""

print(pipe(raw_code)[0]["summary_text"])

# Tokenize the input text
inputs = tokenizer(raw_code, return_tensors="pt")

# Generate decoder input ids
decoder_input_ids = tokenizer(raw_code, return_tensors="pt").input_ids

# Perform inference
with torch.no_grad():
    outputs = model(**inputs, decoder_input_ids=decoder_input_ids)

# Extract the last hidden states
last_hidden_states = outputs.encoder_last_hidden_state

# Calculate the mean of the token embeddings to get a single vector for the sentence
sentence_embedding = torch.mean(last_hidden_states, dim=1)

print("Sentence Embedding:", sentence_embedding)

print("End Time:", datetime.now().strftime("%H:%M:%S"))



# https://towardsdatascience.com/cracking-the-code-llms-354505c53295
# https://medium.com/aimonks/code-llama-quick-start-guide-and-prompt-engineering-eb1de8758399

# Your max_length is set to 200, but your input_length is only 88. Since this is a summarization task, where outputs shorter than the input are typically wanted, you might consider decreasing max_length manually, e.g. summarizer('...', max_length=44)
# 1. Create a function is_prime(n) that will return True if n is a prime number.
# 2. Next step is to check if the n is less than 2 and if it is, it will return False.
# 3. Then we will call the math.sqrt function of n.
# 4. At last step is checking if the number is divisible by 2.
# 5. Then, we will use the range of 2 up to sqrt(n + 1).
# 6. Then it will check for factors from 2 to √n and return True.
# Sentence Embedding: tensor([[0.1505, 0.0178, 0.2757,  ..., 0.0535, 0.1572, 0.0402]])
