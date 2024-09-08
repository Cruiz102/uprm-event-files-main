import openai
from collections import Counter
import csv
import pandas as pd
import re

# Set up your OpenAI API key
client = openai.OpenAI()

# Function to generate synthetic data using OpenAI API
def generate_synthetic_text(category, domain, num_samples=10):
    generated_texts = []

    # Create the prompt
    prompt = """
Prompt:

I have five classifications for a dataset based on abstracts, with a set of commonly associated words for each class. Generate synthetic abstracts for each classification (Civil, MAE, CS, ECE, Psychology), ensuring that the provided common words for each class are naturally included in a coherent way. These abstracts should resemble real-world scientific abstracts and vary in structure, style, and focus. Each abstract should be detailed and extensive, simulating the depth found in academic papers (at least 500 words).

The output should be a Python list of dictionaries with the following structure:
just generate me the list data dont generate me anything else
python
Copiar código
data = [
    {
        "Y1": <integer identifier for classification>,
        "Domain": "<classification domain>",
        "Abstract": "<generated synthetic abstract>"
    },
    # Add more entries for each classification...

Classification Details:

Civil (Y1: 4): system, use, model, effect, differ, develop, design, perform
MAE (Y1: 3): model, use, process, design, study, perform, matter, result
CS (Y1: 0): use, model, data, develop, method, system, result, problem
ECE (Y1: 1): model, control, base, design, parameter
Psychology (Y1: 2): use, associate, result, patient, group, behavior, measure, assess
Important: Do not generate abstracts for the Medical classification.
    """
    
    # Use the OpenAI client to generate text
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        n=num_samples
    )
    
    for choice in response.choices:
        generated_texts.append({
            'Y1': category,
            'Domain': domain,
            'Abstract': choice.message.content.strip()
        })
    
    return generated_texts

def main():
    train_data, valid_data, test_data = preprocess_dataset(train_data, valid_data, test_data)

if __name__ == "__main__":
    main()
