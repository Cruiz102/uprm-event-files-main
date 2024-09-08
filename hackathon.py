import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, confusion_matrix
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
model_name = "allenai/scibert_scivocab_uncased"
checkpoints_folder = ""
output_dir = './scibert2'
# Load the pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define Custom Dataset
class CustomDataset(Dataset):
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
import os

def plot_confusion_matrix(cm, filename="confusion_matrix.png"):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    # Save the plot in the current directory
    current_directory = os.getcwd()  # Get current working directory
    file_path = os.path.join(current_directory, filename)
    plt.savefig(file_path)  # Save the figure
    
    plt.close()  # Close the plot to free up memory
    print(f"Confusion matrix saved to {file_path}")
# Train the model using CUDA (if available)
def train_model(train_dataset, val_dataset, model):
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Make the model tensors contiguous
    def make_model_contiguous(model):
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    make_model_contiguous(model)

    # Check for the latest checkpoint
    last_checkpoint = None
    if os.path.isdir(checkpoints_folder):
        checkpoints = [d for d in os.listdir(checkpoints_folder) if d.startswith('checkpoint')]
        if checkpoints:
            last_checkpoint = os.path.join(checkpoints_folder, sorted(checkpoints)[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")
    # Define the combined training arguments with TensorBoard logging
    training_args = TrainingArguments(
        output_dir=output_dir,                 # Directory to save model and logs
        evaluation_strategy="epoch",           # Evaluate after every epoch
        learning_rate=2e-5,                    # Learning rate
        per_device_train_batch_size=64,        # Batch size for training
        per_device_eval_batch_size=64,         # Batch size for evaluation
        num_train_epochs=20,                   # Number of epochs
        weight_decay=0.01,                     # Weight decay for optimizer
        logging_dir='./logs',                  # Directory for TensorBoard logs
        logging_steps=100,                     # Log every 100 steps
        save_strategy="epoch",                 # Save checkpoint after every epoch
        save_total_limit=2,                    # Keep only the last 2 checkpoints
        fp16=True,                             # Enable mixed precision training (faster on CUDA)
        resume_from_checkpoint=last_checkpoint if last_checkpoint else None,  # Resume from last checkpoint if available
        report_to="tensorboard",               # Report to TensorBoard
        logging_first_step=True,               # Log on the first step
        eval_steps=100,                        # Evaluate every 500 steps (only if needed)
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Start training
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

    # Load the BERT model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)

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
