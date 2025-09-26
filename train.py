import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, EncoderDecoderModel
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import re
import random
from typing import List, Dict, Tuple, Any
import os
from tqdm import tqdm
import argparse
import pickle

class CycleNERDataset(Dataset):
    def __init__(self, sentences: List[str], entity_sequences: List[str], tokenizer, max_length=512):
        self.sentences = sentences
        self.entity_sequences = entity_sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return max(len(self.sentences), len(self.entity_sequences))
    
    def __getitem__(self, idx):
        # Cycle through sentences and entity sequences if lengths differ
        sentence_idx = idx % len(self.sentences)
        entity_idx = idx % len(self.entity_sequences)
        
        sentence = self.sentences[sentence_idx]
        entity_seq = self.entity_sequences[entity_idx]
        
        return {
            'sentence': sentence,
            'entity_sequence': entity_seq
        }

class MaterialsDataProcessor:
    """Process materials science JSON data for CycleNER training"""
    
    def __init__(self):
        # Define material entity types based on your JSON structure
        self.entity_types = {
            'PRECURSOR': 'precursor',
            'TARGET': 'target', 
            'SOLVENT': 'solvent',
            'TEMPERATURE': 'temperature',
            'TIME': 'time',
            'OPERATION': 'operation',
            'QUANTITY': 'quantity'
        }
    
    def extract_entities_from_json(self, data: Dict) -> Tuple[str, List[Tuple[str, str]]]:
        """Extract entities from a single JSON record"""
        paragraph = data.get('paragraph_string', '')
        entities = []
        
        # Extract precursors
        for precursor in data.get('precursors', []) or []:
            material_name = precursor.get('material_string', '').strip()
            if material_name and material_name in paragraph:
                entities.append((material_name, 'PRECURSOR'))
        
        # Extract target materials
        target = data.get('target', {}) or {}
        if target:
            target_name = target.get('material_string', '').strip()
            if target_name and target_name in paragraph:
                entities.append((target_name, 'TARGET'))
        
        # Extract solvents
        for solvent in data.get('solvents_string', []) or []:
            if solvent and solvent in paragraph:
                entities.append((solvent, 'SOLVENT'))
        
        # Extract operations
        for operation in data.get('operations', []) or []:
            if not operation:
                continue
            op_string = operation.get('string', '').strip()
            if op_string and op_string in paragraph:
                entities.append((op_string, 'OPERATION'))
                
                # Extract temperature and time from conditions
                conditions = operation.get('conditions', {}) or {}
                if conditions.get('temperature'):
                    temp_info = conditions['temperature']
                    if isinstance(temp_info, dict) and 'values' in temp_info and temp_info['values']:
                        temp_val = f"{temp_info['values'][0]} {temp_info.get('units', '°C')}"
                        entities.append((temp_val, 'TEMPERATURE'))
                
                if conditions.get('time') :
                    time_info = conditions['time']
                    if isinstance(time_info, dict) and 'values' in time_info and time_info['values']:
                        time_val = f"{time_info['values'][0]} {time_info.get('units', 'h')}"
                        entities.append((time_val, 'TIME'))
        
        # Extract quantities
        for quantity in data.get('quantities', []) or []:
            if not quantity:
                continue
            material = quantity.get('material', '')
            if material and material in paragraph:
                entities.append((material, 'QUANTITY'))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity, etype in entities:
            key = (entity, etype)
            if key not in seen:
                seen.add(key)
                unique_entities.append((entity, etype))
        
        return paragraph, unique_entities
    
    def create_entity_sequence(self, entities: List[Tuple[str, str]]) -> str:
        """Convert list of entities to CycleNER format: entity <sep> type <sep> entity <sep> type"""
        if not entities:
            return ""
        
        sequence_parts = []
        for entity, entity_type in entities:
            sequence_parts.extend([entity.strip(), entity_type])
        
        return " <sep> ".join(sequence_parts)
    
    def process_json_data(self, json_data: List[Dict]) -> Tuple[List[str], List[str]]:
        """Process entire JSON dataset"""
        sentences = []
        entity_sequences = []
        
        for record in json_data:
            paragraph, entities = self.extract_entities_from_json(record)
            if paragraph.strip() and entities:  # Only include if we have both text and entities
                sentences.append(paragraph.strip())
                entity_seq = self.create_entity_sequence(entities)
                entity_sequences.append(entity_seq)
        
        return sentences, entity_sequences
    
    def create_synthetic_entity_sequences(self, entities_list: List[List[Tuple[str, str]]], 
                                        num_synthetic: int = 1000) -> List[str]:
        """Create synthetic entity sequences by combining entities"""
        synthetic_sequences = []
        all_entities_by_type = {}
        
        # Group entities by type
        for entities in entities_list:
            for entity, etype in entities:
                if etype not in all_entities_by_type:
                    all_entities_by_type[etype] = []
                all_entities_by_type[etype].append(entity)
        
        # Remove duplicates
        for etype in all_entities_by_type:
            all_entities_by_type[etype] = list(set(all_entities_by_type[etype]))
        
        # Create synthetic sequences
        for _ in range(num_synthetic):
            # Random sequence length (1-4 entities)
            seq_length = random.randint(1, min(4, len(all_entities_by_type)))
            
            # Select random entity types
            selected_types = random.sample(list(all_entities_by_type.keys()), seq_length)
            
            synthetic_entities = []
            for etype in selected_types:
                if all_entities_by_type[etype]:
                    entity = random.choice(all_entities_by_type[etype])
                    synthetic_entities.append((entity, etype))
            
            if synthetic_entities:
                synthetic_seq = self.create_entity_sequence(synthetic_entities)
                synthetic_sequences.append(synthetic_seq)
        
        return synthetic_sequences

class CycleNER:
    def __init__(self, model_name="t5-small", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens = {"additional_special_tokens": ["<sep>"],
                          "pad_token": "<pad>"
                          }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Initialize S2E and E2S models
        self.s2e_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        model_name, model_name).to(device)
        self.e2s_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        model_name, model_name).to(device)
        
        # Configure the encoder-decoder models
        self.s2e_model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.s2e_model.config.eos_token_id = self.tokenizer.eos_token_id
        self.s2e_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Same for e2s_model
        self.e2s_model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.e2s_model.config.eos_token_id = self.tokenizer.eos_token_id
        self.e2s_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Resize embeddings for new tokens
        self.s2e_model.resize_token_embeddings(len(self.tokenizer))
        self.e2s_model.resize_token_embeddings(len(self.tokenizer))
        
        # Optimizers
        self.s2e_optimizer = optim.Adam(self.s2e_model.parameters(), lr=5e-5)
        self.e2s_optimizer = optim.Adam(self.e2s_model.parameters(), lr=5e-5)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
    def encode_batch(self, texts: List[str], max_length: int = 512):
        """Encode batch of texts"""
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def decode_batch(self, token_ids: torch.Tensor) -> List[str]:
        """Decode batch of token IDs"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    def s2e_forward(self, sentences: List[str], batch_size: int = 8) -> List[str]:
        """Sentence-to-Entity forward pass with batching"""
        if not sentences:
            return []
            
        all_predictions = []
        
        # Process in batches to avoid OOM
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
            # Add task prefix for T5
            prefixed_sentences = [f"extract entities: {sent}" for sent in batch_sentences]
            
            inputs = self.encode_batch(prefixed_sentences)
            
            with torch.no_grad():
                outputs = self.s2e_model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=2,
                    temperature=1.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            batch_predictions = self.decode_batch(outputs)
            all_predictions.extend(batch_predictions)
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()
        
        return all_predictions
    
    def e2s_forward(self, entity_sequences: List[str], batch_size: int = 8) -> List[str]:
        """Entity-to-Sentence forward pass with batching"""
        if not entity_sequences:
            return []
            
        all_predictions = []
        
        # Process in batches to avoid OOM
        for i in range(0, len(entity_sequences), batch_size):
            batch_sequences = entity_sequences[i:i+batch_size]
            
            # Add task prefix for T5
            prefixed_sequences = [f"generate sentence: {seq}" for seq in batch_sequences]
            
            inputs = self.encode_batch(prefixed_sequences)
            
            with torch.no_grad():
                outputs = self.e2s_model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=2,
                    temperature=1.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            batch_predictions = self.decode_batch(outputs)
            all_predictions.extend(batch_predictions)
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()
        
        return all_predictions
    
    def train_s2e(self, sentences: List[str], target_sequences: List[str]):
        """Train S2E model"""
        self.s2e_model.train()
        
        prefixed_sentences = [f"extract entities: {sent}" for sent in sentences]
        inputs = self.encode_batch(prefixed_sentences)
        targets = self.encode_batch(target_sequences)
        
        outputs = self.s2e_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=targets['input_ids']
        )
        
        loss = outputs.loss
        loss.backward()
        self.s2e_optimizer.step()
        self.s2e_optimizer.zero_grad()
        
        return loss.item()
    
    def train_e2s(self, entity_sequences: List[str], target_sentences: List[str]):
        """Train E2S model"""
        self.e2s_model.train()
        
        prefixed_sequences = [f"generate sentence: {seq}" for seq in entity_sequences]
        inputs = self.encode_batch(prefixed_sequences)
        targets = self.encode_batch(target_sentences)
        
        outputs = self.e2s_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=targets['input_ids']
        )
        
        loss = outputs.loss
        loss.backward()
        self.e2s_optimizer.step()
        self.e2s_optimizer.zero_grad()
        
        return loss.item()
    
    def cycle_training_step(self, sentences: List[str], entity_sequences: List[str]):
        """Perform one cycle training step"""
        
        # Use smaller inference batch size for memory efficiency
        inference_batch_size = 4
        
        # S-cycle: S -> S2E -> E2S -> S'
        synthetic_entity_seqs = self.s2e_forward(sentences, batch_size=inference_batch_size)  # S2E(S) -> Q'
        s_cycle_loss = self.train_e2s(synthetic_entity_seqs, sentences)  # Train E2S with (Q', S)
        
        # E-cycle: Q -> E2S -> S2E -> Q'
        synthetic_sentences = self.e2s_forward(entity_sequences, batch_size=inference_batch_size)  # E2S(Q) -> S'
        e_cycle_loss = self.train_s2e(synthetic_sentences, entity_sequences)  # Train S2E with (S', Q)
        
        return s_cycle_loss, e_cycle_loss
    
    def evaluate_ner(self, test_sentences: List[str], true_entity_sequences: List[str], eval_batch_size: int = 16) -> Dict:
        """Evaluate NER performance with batching to avoid OOM"""
        self.s2e_model.eval()
        
        predicted_sequences = []
        
        print(f"Evaluating on {len(test_sentences)} samples in batches of {eval_batch_size}...")
        
        # Process in smaller batches to avoid OOM
        for i in tqdm(range(0, len(test_sentences), eval_batch_size), desc="Validation"):
            batch_sentences = test_sentences[i:i+eval_batch_size]
            
            # Clear cache before each batch
            torch.cuda.empty_cache()
            
            batch_predictions = self.s2e_forward(batch_sentences)
            predicted_sequences.extend(batch_predictions)
        
        # Simple evaluation - you might want to implement more sophisticated metrics
        exact_matches = sum(1 for pred, true in zip(predicted_sequences, true_entity_sequences) 
                           if pred.strip() == true.strip())
        
        accuracy = exact_matches / len(test_sentences) if test_sentences else 0.0
        
        return {
            'accuracy': accuracy,
            'exact_matches': exact_matches,
            'total_samples': len(test_sentences)
        }
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int):
        """Save complete checkpoint including model states, optimizer states, and training info"""
        os.makedirs(checkpoint_path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            's2e_model_state': self.s2e_model.state_dict(),
            'e2s_model_state': self.e2s_model.state_dict(),
            's2e_optimizer_state': self.s2e_optimizer.state_dict(),
            'e2s_optimizer_state': self.e2s_optimizer.state_dict(),
            'tokenizer_vocab_size': len(self.tokenizer)
        }
        
        # Save checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_path, 'checkpoint.pt'))
        
        # Save tokenizer separately (for easy loading)
        self.tokenizer.save_pretrained(os.path.join(checkpoint_path, "tokenizer"))
        
        print(f"Checkpoint saved to {checkpoint_path} at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str, model_name: str = "t5-small"):
        """Load checkpoint and resume training"""
        checkpoint_file = os.path.join(checkpoint_path, 'checkpoint.pt')
        tokenizer_path = os.path.join(checkpoint_path, "tokenizer")
        
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load tokenizer
        if os.path.exists(tokenizer_path):
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        else:
            print("Warning: Tokenizer not found in checkpoint, using default")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            special_tokens = {"additional_special_tokens": ["<sep>"]}
            self.tokenizer.add_special_tokens(special_tokens)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # Restore training state
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        # Initialize models with correct vocab size
        self.s2e_model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.e2s_model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.s2e_model.resize_token_embeddings(len(self.tokenizer))
        self.e2s_model.resize_token_embeddings(len(self.tokenizer))
        
        # Load model states
        self.s2e_model.load_state_dict(checkpoint['s2e_model_state'])
        self.e2s_model.load_state_dict(checkpoint['e2s_model_state'])
        
        # Initialize optimizers
        self.s2e_optimizer = optim.Adam(self.s2e_model.parameters(), lr=5e-5)
        self.e2s_optimizer = optim.Adam(self.e2s_model.parameters(), lr=5e-5)
        
        # Load optimizer states
        self.s2e_optimizer.load_state_dict(checkpoint['s2e_optimizer_state'])
        self.e2s_optimizer.load_state_dict(checkpoint['e2s_optimizer_state'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch
    
    def train(self, train_sentences: List[str], train_entity_sequences: List[str],
              val_sentences: List[str] = None, val_entity_sequences: List[str] = None,
              epochs: int = 10, batch_size: int = 8, checkpoint_path: str = None, 
              save_every: int = 5, resume_from_checkpoint: str = None):
        """Train CycleNER model with checkpoint support"""
        
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            try:
                start_epoch = self.load_checkpoint(resume_from_checkpoint)
                print(f"Resumed training from epoch {start_epoch}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting fresh training...")
                start_epoch = 0
        
        # Create checkpoint directory if specified
        if checkpoint_path:
            os.makedirs(checkpoint_path, exist_ok=True)
        
        for epoch in range(start_epoch, epochs):
            epoch_s_losses = []
            epoch_e_losses = []
            
            # Shuffle data
            combined = list(zip(train_sentences, train_entity_sequences))
            random.shuffle(combined)
            train_sentences_shuffled, train_entity_sequences_shuffled = zip(*combined)
            
            # Training batches
            num_batches = (len(train_sentences_shuffled) + batch_size - 1) // batch_size
            
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
            
            for i in progress_bar:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(train_sentences_shuffled))
                
                batch_sentences = list(train_sentences_shuffled[start_idx:end_idx])
                batch_entity_seqs = list(train_entity_sequences_shuffled[start_idx:end_idx])
                
                s_loss, e_loss = self.cycle_training_step(batch_sentences, batch_entity_seqs)
                
                epoch_s_losses.append(s_loss)
                epoch_e_losses.append(e_loss)
                
                progress_bar.set_postfix({
                    'S-Loss': f'{s_loss:.4f}',
                    'E-Loss': f'{e_loss:.4f}'
                })
            
            avg_s_loss = np.mean(epoch_s_losses)
            avg_e_loss = np.mean(epoch_e_losses)
            
            # Update training state
            self.current_epoch = epoch
            
            # Validation
            val_metrics = {}
            if val_sentences and val_entity_sequences:
                val_metrics = self.evaluate_ner(val_sentences, val_entity_sequences)
                
                # Use E-cycle loss as stopping criterion (as per paper)
                if avg_e_loss < self.best_val_loss:
                    self.best_val_loss = avg_e_loss
                    if checkpoint_path:
                        self.save_models(os.path.join(checkpoint_path, f"best_model_epoch_{epoch}"))
            
            # Save training history
            epoch_info = {
                'epoch': epoch + 1,
                'avg_s_loss': avg_s_loss,
                'avg_e_loss': avg_e_loss,
                **val_metrics
            }
            self.training_history.append(epoch_info)
            
            print(f"Epoch {epoch+1}: S-Loss={avg_s_loss:.4f}, E-Loss={avg_e_loss:.4f}")
            if val_metrics:
                print(f"Validation Accuracy: {val_metrics.get('accuracy', 0):.4f}")
            
            # Save checkpoint periodically
            if checkpoint_path and (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch+1}"), 
                    epoch
                )
        
        # Save final checkpoint
        if checkpoint_path:
            self.save_checkpoint(os.path.join(checkpoint_path, "final_checkpoint"), epochs - 1)
            
        return self.training_history
    
    def save_models(self, save_path: str):
        """Save both S2E and E2S models"""
        os.makedirs(save_path, exist_ok=True)
        
        self.s2e_model.save_pretrained(os.path.join(save_path, "s2e_model"))
        self.e2s_model.save_pretrained(os.path.join(save_path, "e2s_model"))
        self.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
        
        print(f"Models saved to {save_path}")
    
    def load_models(self, load_path: str):
        """Load both S2E and E2S models"""
        self.s2e_model = T5ForConditionalGeneration.from_pretrained(
            os.path.join(load_path, "s2e_model")
        ).to(self.device)
        
        self.e2s_model = T5ForConditionalGeneration.from_pretrained(
            os.path.join(load_path, "e2s_model")
        ).to(self.device)
        
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(load_path, "tokenizer"))
        
        print(f"Models loaded from {load_path}")

def main():
    parser = argparse.ArgumentParser(description='Train CycleNER on Materials Science Data')
    parser.add_argument('--data_path', type=str, default="solution-synthesis_dataset_2021-8-5.json", help='Path to JSON data file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--model_name', type=str, default='t5-small', help='T5 model variant')
    parser.add_argument('--save_path', type=str, default='./cyclener_models', help='Model save path')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic entity sequences')
    parser.add_argument('--synthetic_count', type=int, default=1000, help='Number of synthetic sequences')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='Checkpoint save path')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Load JSON data
    print(f"Loading data from {args.data_path}")
    with open(args.data_path, 'r') as f:
        json_data = json.load(f)
    
    # Process data
    processor = MaterialsDataProcessor()
    sentences, entity_sequences = processor.process_json_data(json_data)
    
    print(f"Processed {len(sentences)} sentences and {len(entity_sequences)} entity sequences")
    
    # Create synthetic sequences if requested
    if args.use_synthetic:
        print("Creating synthetic entity sequences...")
        all_entities = []
        for record in json_data:
            _, entities = processor.extract_entities_from_json(record)
            if entities:
                all_entities.append(entities)
        
        synthetic_sequences = processor.create_synthetic_entity_sequences(
            all_entities, args.synthetic_count
        )
        entity_sequences.extend(synthetic_sequences)
        print(f"Added {len(synthetic_sequences)} synthetic sequences")
    
    # Split data (80% train, 20% validation)
    split_idx = int(0.8 * len(sentences))
    train_sentences = sentences[:split_idx]
    train_entity_sequences = entity_sequences[:split_idx]
    val_sentences = sentences[split_idx:]
    val_entity_sequences = entity_sequences[split_idx:]
    
    print(f"Training data: {len(train_sentences)} sentences, {len(train_entity_sequences)} entity sequences")
    print(f"Validation data: {len(val_sentences)} sentences, {len(val_entity_sequences)} entity sequences")
    
    # Initialize model
    print("Initializing CycleNER...")
    cyclener = CycleNER(model_name=args.model_name)
    
    print("Starting training...")
    history = cyclener.train(
        train_sentences=train_sentences,
        train_entity_sequences=train_entity_sequences,
        val_sentences=val_sentences,
        val_entity_sequences=val_entity_sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        save_every=args.save_every,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Save final models
    cyclener.save_models(args.save_path)
    
    # Save training history
    with open(os.path.join(args.save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("Training completed!")
    
    # Test on a few examples
    print("\nTesting on sample sentences...")
    test_samples = val_sentences[:5]
    predicted_entities = cyclener.s2e_forward(test_samples)
    
    for i, (sent, pred) in enumerate(zip(test_samples, predicted_entities)):
        print(f"\nExample {i+1}:")
        print(f"Sentence: {sent[:100]}...")
        print(f"Predicted Entities: {pred}")

if __name__ == "__main__":
    main()