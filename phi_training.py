from openai import OpenAI
import csv
import pandas as pd




import csv
import re

# Function to clean and filter out non-plain text content from abstracts# Function to extract text within ```plaintext``` blocks
def extract_plaintext(text):
    # Regular expression to extract content within ```plaintext``` blocks
    matches = re.findall(r'```plaintext(.*?)```', text, re.DOTALL)
    # Join all matches (if there are multiple blocks of ```plaintext```)
    extracted_text = ' '.join([match.strip() for match in matches])
    return extracted_text
# Function to parse the CSV and clean the plaintext in the 'Abstract' column
def parse_and_filter_csv(input_csv, output_csv):
    with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames  # Keep the same fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for row in reader:
            # Clean the 'Abstract' column
            row['Abstract'] = extract_plaintext(row['Abstract'])
            writer.writerow(row)

    print(f"Filtered data saved to {output_csv}")

# Function to generate synthetic data using OpenAI API
def generate_synthetic_text(category, domain, num_samples=10):
    generated_texts = []

    # Create the prompt
    prompt = f"""
I have five classifications for a dataset based on abstracts, with a set of commonly associated words for each class. Generate synthetic abstracts for each classification (Civil, MAE, CS, ECE, Psychology), ensuring that the provided common words for each class are naturally included in a coherent way. These abstracts should resemble real-world scientific abstracts and vary in structure, style, and focus. Each abstract should be detailed and extensive, simulating the depth found in academic papers (at least 500 words).

The output should be a Python list of dictionaries with the following structure:
data = [
    {{
        "Y1": {category},
        "Domain": "{domain}",
        "Abstract": "<generated synthetic abstract>"
    }},
    # Add more entries for each classification...
]
    """
    client = OpenAI()
    # Use the OpenAI API to generate the abstract
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        n=num_samples
    )

    # Parse response
    for choice in response.choices:
        generated_texts.append({
            'Y1': category,
            'Domain': domain,
            'Abstract': choice.message.content.strip()
        })

    return generated_texts

# Function to format data and save it to a CSV file
def save_to_csv(data, filename='generated_abstracts.csv'):
    # Define the CSV file fieldnames
    fieldnames = ['Y1', 'Domain', 'Abstract']
    
    # Write the data to a CSV file
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Data successfully saved to {filename}")

# Main function to generate synthetic abstracts and save to CSV
def main():
    # Define the classifications and associated Y1 values
    classifications = [
        {'category': 4, 'domain': 'Civil'},
        {'category': 3, 'domain': 'MAE'},
        {'category': 0, 'domain': 'CS'},
        {'category': 1, 'domain': 'ECE'},
        {'category': 2, 'domain': 'Psychology'}
    ]
    
    # # Collect generated abstracts for each classification
    # all_data = []
    # for cls in classifications:
    #     data = generate_synthetic_text(cls['category'], cls['domain'], num_samples=5)
    #     all_data.extend(data)

    # # Save the generated data to CSV
    # save_to_csv(all_data)
    input_csv = 'synthetic_data.csv'  # Input CSV with generated abstracts
    output_csv = 'filtered_abstracts.csv'  # Output CSV after filtering plaintext
    
    # Parse and filter the CSV
    parse_and_filter_csv(input_csv, output_csv)



if __name__ == "__main__":
    main()
