import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# Load your dataset (ensure you have uploaded it to Kaggle)
data = pd.read_csv('/kaggle/input/sentiment-analysis-for-financial-news/all-data.csv', encoding='latin1', header=None)
data.columns = ['Sentiment', 'Sentence']

print("Dataset loaded successfully.")
print(f"Dataset shape: {data.shape}")

# Preprocess the data
data.dropna(subset=['Sentiment', 'Sentence'], inplace=True)
print("Dropped rows with missing values.")
print(f"Dataset shape after dropping missing values: {data.shape}")

# Map sentiment labels to numeric values
sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
data['Sentiment'] = data['Sentiment'].map(sentiment_mapping)
print("Mapped sentiment labels to numeric values.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Sentence'], data['Sentiment'], test_size=0.2, random_state=42)
print("Split the data into training and testing sets.")
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Initialized BERT tokenizer.")

# Tokenize the data
print("Tokenizing training data...")
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
print("Tokenizing testing data...")
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)
print("Tokenization complete.")

# Convert labels to tensors
train_labels = torch.tensor(y_train.values)
test_labels = torch.tensor(y_test.values)

# Convert encodings to tensors
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']), train_labels)
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']), torch.tensor(test_encodings['attention_mask']),
                             test_labels)

# Initialize BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
print("Initialized BERT model and optimizer.")


# Training function
def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d[0].to(device)
        attention_mask = d[1].to(device)
        labels = d[2].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


# Evaluation function
def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d[0].to(device)
            attention_mask = d[1].to(device)
            labels = d[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            logits = outputs[1]

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
epochs = 10
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Training loop
print("Starting training loop...")
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')

    val_acc, val_loss = eval_model(model, test_loader, device)
    print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}')

# Save the model
model_save_path = '/kaggle/working/bert_sentiment_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

print("Training complete.")