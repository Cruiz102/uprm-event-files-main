import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PhiForTokenClassification, PhiConfig, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import f1_score, confusion_matrix
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os

phi_model = "microsoft/phi-2"
# Load the pretrained Phi tokenizer
tokenizer = AutoTokenizer.from_pretrained(phi_model)
# Add a pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Define Custom Dataset
# Define Custom Dataset for sequence-level labels
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels  # Single label for each sequence
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]  # This is a single integer label, not a list

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
            'labels': torch.tensor(label, dtype=torch.long)  # Single label, no list
        }
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

# Prepare datasets
def prepare_datasets(train_data, valid_data, test_data, tokenizer):
    X_train = train_data['Abstract'].values
    y_train = train_data['Y1'].values

    X_val = valid_data['Abstract'].values
    y_val = valid_data['Y1'].values

    X_test = test_data['Abstract'].values
    y_test = test_data['Y1'].values

    train_dataset = CustomDataset(X_train, y_train, tokenizer)
    val_dataset = CustomDataset(X_val, y_val, tokenizer)
    test_dataset = CustomDataset(X_test, y_test, tokenizer)

    return train_dataset, val_dataset, test_dataset

# Define the metrics computation function
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    f1 = f1_score(labels, pred, average='weighted')
    cm = confusion_matrix(labels, pred)
    
    # Convert confusion matrix to a list for JSON serialization
    return {
        'f1': f1,
        'confusion_matrix': cm.tolist()  # Convert NumPy array to list
    }

# Define confusion matrix plotting
def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Load the Phi model for token classification
def load_phi_model():
    # Load the Phi model configuration
    config = PhiConfig.from_pretrained("microsoft/phi-2")

    # Load the Phi model for token classification
    model = PhiForTokenClassification.from_pretrained("microsoft/phi-2", config=config)

    return model

# Train the model using CUDA (if available)
def train_model(train_dataset, val_dataset, model):
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Check for the latest checkpoint
    last_checkpoint = None
    if os.path.isdir('./results'):
        checkpoints = [d for d in os.listdir('./results') if d.startswith('checkpoint')]
        if checkpoints:
            last_checkpoint = os.path.join('./results', sorted(checkpoints)[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        fp16=True,  # Enable mixed precision training for faster training on CUDA
        save_strategy="epoch",  # Save checkpoint after every epoch
        resume_from_checkpoint=last_checkpoint if last_checkpoint else None,  # Resume from checkpoint if available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)
    return trainer

# Main function
def main():
    # Load and preprocess data
    train_data, valid_data, test_data = load_data(
        'data/train_data.csv',
        'data/valid_data.csv',
        'data/test_data.csv'
    )

    train_data, valid_data, test_data = preprocess_dataset(train_data, valid_data, test_data)

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(train_data, valid_data, test_data, tokenizer)

    # Load the Phi model for token classification
    model = load_phi_model()

    # Train the model
    trainer = train_model(train_dataset, val_dataset, model)

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Perform manual prediction to compute the confusion matrix
    predictions = trainer.predict(val_dataset)
    metrics = compute_metrics((predictions.predictions, predictions.label_ids))

    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])

# Run the main function
if __name__ == "__main__":
    main()
