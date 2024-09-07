import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

model_name = "allenai/scibert_scivocab_uncased"
checkpoint = 'scibert/checkpoint-3070'

# Load the pretrained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define Custom Dataset for Test Data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Preprocessing function for text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# Load dataset
def load_test_data(test_path):
    test_data = pd.read_csv(test_path)
    return test_data

# Preprocess data
def preprocess_test_dataset(test_data):
    test_data['Abstract'] = test_data['Abstract'].apply(preprocess_text)
    return test_data

# Prepare datasets
def prepare_test_dataset(test_data, tokenizer):
    X_test = test_data['Abstract'].values
    y_test = test_data['Y1'].values
    test_dataset = CustomDataset(X_test, y_test, tokenizer)
    return test_dataset, y_test

# Define the metrics computation function
def compute_metrics(predictions, labels):
    pred = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, pred, average='weighted')
    cm = confusion_matrix(labels, pred)
    return {'f1': f1, 'confusion_matrix': cm}

# Define confusion matrix plotting
def plot_confusion_matrix(cm, model_name, checkpoint, filename="confusion_matrix2.png"):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {model_name} at {checkpoint}')
    
    # Save the plot in the current directory
    current_directory = os.getcwd()  # Get current working directory
    file_path = os.path.join(current_directory, filename)
    plt.savefig(file_path)  # Save the figure
    
    plt.close()  # Close the plot to free up memory
    print(f"Confusion matrix saved to {file_path}")

def test_model():
    # Load the pre-trained model
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=6)

    model.eval()  # Put model in evaluation mode

    # Load and preprocess test data
    test_data = load_test_data('data/test_data.csv')
    test_data = preprocess_test_dataset(test_data)

    # Prepare test dataset
    test_dataset, y_test = prepare_test_dataset(test_data, tokenizer)

    # Initialize Trainer
    trainer = Trainer(model=model)

    # Make predictions on test set
    predictions = trainer.predict(test_dataset)

    # Compute metrics
    metrics = compute_metrics(predictions.predictions, y_test)

    # Print F1-score
    print(f"F1-score: {metrics['f1']}")

    # Plot confusion matrix with model name and checkpoint
    plot_confusion_matrix(metrics['confusion_matrix'], model_name, checkpoint)

if __name__ == "__main__":
    test_model()
