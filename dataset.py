import openai
from collections import Counter
import csv
import pandas as pd
import re

# Set up your OpenAI API key
client = openai.OpenAI()



# Preprocessing function for text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# Load dataset
def load_data(train_path, valid_path, test_path):
    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    test_data = pd.read_csv(test_path)
    return train_data, valid_data, test_data

# Preprocess data
def preprocess_dataset(train_data, valid_data, test_data):
    train_data['Abstract'] = train_data['Abstract'].apply(preprocess_text)
    valid_data['Abstract'] = valid_data['Abstract'].apply(preprocess_text)
    test_data['Abstract'] = test_data['Abstract'].apply(preprocess_text)
    return train_data, valid_data, test_data




# Function to analyze common words for each category
def get_most_common_words_per_category(data, text_column, label_column, n=10):
    category_words = {}
    
    # Iterate over each category
    for category in data[label_column].unique():
        # Filter data by category
        category_texts = data[data[label_column] == category][text_column]
        
        # Tokenize and count word frequency
        words = []
        for text in category_texts:
            words.extend(text.split())
        
        # Count most common words
        common_words = Counter(words).most_common(n)
        category_words[category] = [word for word, _ in common_words]
    
    return category_words

# Function to generate synthetic data using OpenAI API
def generate_synthetic_text(category, common_words, domain, num_samples=10):
    generated_texts = []
    print(common_words)
    
    # # Create the prompt
    # prompt = f"Generate {num_samples} abstracts for the category '{category}' using these common words: {', '.join(common_words)}."
    
    # # Use the OpenAI client to generate text
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ],
    #     n=num_samples
    # )
    
    # for choice in response.choices:
    #     generated_texts.append({
    #         'Y1': category,
    #         'Domain': domain,
    #         'Abstract': choice.message.content.strip()
    #     })
    
    # return generated_texts

# Function to create augmented CSV
def create_augmented_csv(data, text_column, label_column, domain_column, output_file, num_samples_per_category=5):
    # Get common words for each category
    common_words_per_category = get_most_common_words_per_category(data, text_column, label_column)
    
    # Generate synthetic data
    synthetic_data = []
    for category, common_words in common_words_per_category.items():
        domain = data[data[label_column] == category][domain_column].iloc[0]
        generated_texts = generate_synthetic_text(category, common_words, domain, num_samples_per_category)
        synthetic_data.extend(generated_texts)
    
    # Write the synthetic data to a new CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Y1', 'Domain', 'Abstract'])
        writer.writeheader()
        writer.writerows(synthetic_data)
    
    print(f"Synthetic data saved to {output_file}")

# Example usage in your workflow
def main():
    # Load and preprocess data
    train_data, valid_data, test_data = load_data(
        'data/train_data.csv',
        'data/valid_data.csv',
        'data/test_data.csv'
    )

    train_data, valid_data, test_data = preprocess_dataset(train_data, valid_data, test_data)

    # Create augmented CSV with synthetic data only
    create_augmented_csv(train_data, text_column='Abstract', label_column='Y1', domain_column='Domain', output_file='synthetic_data.csv')

if __name__ == "__main__":
    main()
