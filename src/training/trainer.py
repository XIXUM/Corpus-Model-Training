import torch
import benepar
from benepar.integrations.downloader import load_trained_model
import nltk
import time
from typing import List, Tuple, Optional
import os
from ..utils.config_loader import load_config, get_device

class InputExample:
    """
    A simple implementation of the input example expected by Benepar.
    """
    def __init__(self, tree: nltk.Tree):
        self.tree = tree
        # Extract words and spaces from the tree leaves
        # This is a simplification; usually we need raw text + alignment.
        # Here we assume the tree leaves ARE the words.
        self._words = tree.leaves()
        self._pos = tree.pos()
        # Assume standard spacing (True) for all except maybe last? 
        # For robust training, one should preserve original spacing.
        # We'll default to True for all.
        self.space_after = [True] * len(self._words)

    @property
    def words(self) -> List[str]:
        return self._words

    def leaves(self) -> List[str]:
        return self._words

    def pos(self) -> List[Tuple[str, str]]:
        return self._pos

class BeneparTrainer:
    def __init__(self, config_path="config/training_config.yaml"):
        self.config = load_config(config_path)
        self.device = get_device(self.config)
        self.model_name = "benepar_en3"
        
        print(f"Loading Benepar model '{self.model_name}' for training...")
        try:
            # Load the pretrained model
            self.parser = load_trained_model(self.model_name)
            self.parser.to(self.device)
            self.parser.train() # Set to training mode
            print("✅ Model loaded and set to training mode.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def train(self, train_data_path: str):
        """
        Runs the training loop on the provided data.
        train_data_path: Path to a file containing bracketed parse trees (PTB format).
        """
        print(f"Starting training on data from: {train_data_path}")
        
        # Load data
        examples = []
        try:
            with open(train_data_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Assume one tree per line or standard PTB format
                # We can use nltk.CorpusReader or simple string parsing
                # Let's try simple bracket parsing if it's a clean file
                # or nltk.Tree.fromstring
                
                # Split by lines, filter empty
                lines = [l.strip() for l in content.split('\n') if l.strip()]
                
                for line in lines:
                    try:
                        tree = nltk.Tree.fromstring(line)
                        examples.append(InputExample(tree))
                    except ValueError:
                        continue # Skip invalid lines
                        
            print(f"Loaded {len(examples)} training examples.")
        except Exception as e:
            print(f"Error loading training data: {e}")
            return

        if not examples:
            print("No valid training examples found.")
            return

        # Setup Optimizer
        learning_rate = self.config.get("training", {}).get("learning_rate", 1e-5)
        optimizer = torch.optim.Adam(self.parser.parameters(), lr=learning_rate)
        
        epochs = self.config.get("training", {}).get("epochs", 3)
        batch_size = self.config.get("training", {}).get("batch_size", 8)
        subbatch_max_tokens = 500 # Standard benepar default
        
        print(f"Training for {epochs} epochs, batch size {batch_size}...")
        
        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0
            num_batches = 0
            
            # Shuffle examples
            import random
            random.shuffle(examples)
            
            # Batching
            for i in range(0, len(examples), batch_size):
                batch_examples = examples[i : i + batch_size]
                
                # Prepare batch
                # Benepar handles collation internally via encode_and_collate_subbatches
                try:
                    # This returns a list of subbatches (dictionaries)
                    subbatches = self.parser.encode_and_collate_subbatches(
                        batch_examples, subbatch_max_tokens=subbatch_max_tokens
                    )
                    
                    batch_loss = 0
                    optimizer.zero_grad()
                    
                    for batch_size_in_subbatch, subbatch in subbatches:
                        # Move subbatch to device? encode_and_collate might not do it for all tensors
                        # ChartParser.forward handles .to(device) for some keys, but let's ensure compatibility
                        # The parser seems to handle device placement in forward/compute_loss if self.device is set.
                        
                        loss = self.parser.compute_loss(subbatch)
                        loss.backward()
                        batch_loss += loss.item()
                    
                    optimizer.step()
                    total_loss += batch_loss
                    num_batches += 1
                    
                    if num_batches % 10 == 0:
                        print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {batch_loss:.4f}")
                        
                except Exception as e:
                    print(f"Error in batch {i//batch_size}: {e}")
                    continue

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}. Time: {time.time() - start_time:.2f}s")
            
            # Save checkpoint
            checkpoint_dir = self.config.get("training", {}).get("checkpoint_dir", "checkpoints")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            
            save_path = os.path.join(checkpoint_dir, f"benepar_epoch_{epoch+1}.pt")
            # We should save the state dict
            try:
                torch.save(self.parser.state_dict(), save_path)
                print(f"Checkpoint saved to {save_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

        print("Training complete.")

