import argparse
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train(epochs, learning_rate, hidden_size):
    print(f"Starting training with {epochs} epochs, lr={learning_rate}, hidden_size={hidden_size}")
    
    # Dummy data for demonstration
    # Sentence: "hello world" -> simple bag of words or random vector
    input_size = 10
    output_size = 2
    
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Dummy training loop
    for epoch in range(epochs):
        # Create random input and target
        inputs = torch.randn(1, input_size)
        labels = torch.tensor([1]) # Dummy class 1
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")
    
    # Example NLTK usage
    text = "Training neural networks with NLTK preprocessing is fun."
    tokens = word_tokenize(text)
    print(f"\nExample NLTK tokenization:\nInput: '{text}'\nTokens: {tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a simple neural network.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=20, help='Hidden layer size')
    
    args = parser.parse_args()
    
    train(args.epochs, args.lr, args.hidden)

