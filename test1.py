import json
import torch
import re
from train import CycleNER, MaterialsDataProcessor

class FixedMaterialsDataProcessor(MaterialsDataProcessor):
    """Extended processor with robust parsing for model outputs"""
    
    def parse_model_output(self, sequence: str) -> list:
        """
        Robust parser that handles multiple output formats from the model
        """
        if not sequence or not sequence.strip():
            return []
        
        sequence = sequence.strip()
        entities = []
        
        # Define known entity types
        entity_types = {'PRECURSOR', 'TARGET', 'SOLVENT', 'TEMPERATURE', 'TIME', 'OPERATION', 'QUANTITY'}
        
        # Method 1: Standard <sep> format
        if " <sep> " in sequence:
            return self.parse_entity_sequence(sequence)
        
        # Method 2: Space-separated format (what model actually produces)
        tokens = sequence.split()
        i = 0
        
        while i < len(tokens):
            # Look for entity type in next few positions
            entity_found = False
            
            # Try 1-word entity + type
            if i < len(tokens) - 1 and tokens[i + 1] in entity_types:
                entity = tokens[i]
                entity_type = tokens[i + 1]
                entities.append((entity, entity_type))
                i += 2
                entity_found = True
                
            # Try 2-word entity + type
            elif i < len(tokens) - 2 and tokens[i + 2] in entity_types:
                entity = f"{tokens[i]} {tokens[i + 1]}"
                entity_type = tokens[i + 2]
                entities.append((entity, entity_type))
                i += 3
                entity_found = True
                
            # Try 3-word entity + type (for chemical formulas like "25.0 °C")
            elif i < len(tokens) - 3 and tokens[i + 3] in entity_types:
                entity = f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
                entity_type = tokens[i + 3]
                entities.append((entity, entity_type))
                i += 4
                entity_found = True
                
            # Try 4-word entity + type (for complex chemical names)
            elif i < len(tokens) - 4 and tokens[i + 4] in entity_types:
                entity = f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]} {tokens[i + 3]}"
                entity_type = tokens[i + 4]
                entities.append((entity, entity_type))
                i += 5
                entity_found = True
            
            if not entity_found:
                i += 1
        
        # Method 3: Pattern-based extraction for chemical formulas
        if not entities:
            # Pattern for chemical formulas: word(s) followed by entity type
            pattern = r'([A-Za-z0-9\(\)\.\-]+(?:\s+[A-Za-z0-9\(\)\.\-°]+)*)\s+(PRECURSOR|TARGET|SOLVENT|TEMPERATURE|TIME|OPERATION|QUANTITY)'
            matches = re.findall(pattern, sequence)
            entities = [(match[0].strip(), match[1].strip()) for match in matches]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity, etype in entities:
            key = (entity, etype)
            if key not in seen and entity and etype:
                seen.add(key)
                unique_entities.append((entity, etype))
        
        return unique_entities
    
    def debug_parsing(self, sequence: str, sample_id: int = None):
        """Debug parsing to see what's happening"""
        if sample_id is not None:
            print(f"\n=== Debug Parsing Sample {sample_id} ===")
        
        print(f"Input sequence: '{sequence}'")
        print(f"Has <sep>: {' <sep> ' in sequence}")
        
        # Try different parsing methods
        try:
            method1 = self.parse_entity_sequence(sequence)
            print(f"Method 1 (original): {method1}")
        except Exception as e:
            print(f"Method 1 failed: {e}")
            method1 = []
        
        try:
            method2 = self.parse_model_output(sequence)
            print(f"Method 2 (robust): {method2}")
        except Exception as e:
            print(f"Method 2 failed: {e}")
            method2 = []
        
        return method2 if method2 else method1

def evaluate_only(checkpoint_path, data_path, model_name='t5-base'):
    """Load trained model and evaluate on test data with fixed parsing"""
    
    # Load data
    print(f"Loading data from {data_path}")
    with open(data_path, 'r') as f:
        json_data = json.load(f)
    
    # Process data with fixed processor
    processor = FixedMaterialsDataProcessor()
    sentences, entity_sequences = processor.process_json_data(json_data)
    
    # Split data (same as training: 80% train, 20% test)
    split_idx = int(0.8 * len(sentences))
    test_sentences = sentences[split_idx:]
    test_entity_sequences = entity_sequences[split_idx:]
    
    print(f"Test data: {len(test_sentences)} sentences")
    
    # Initialize model
    cyclener = CycleNER(model_name=model_name)
    
    # Load from checkpoint
    print(f"Loading model from {checkpoint_path}")
    try:
        cyclener.load_checkpoint(checkpoint_path, model_name)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
    
    # Evaluate on full test set
    print("Running evaluation on full test set...")
    
    # Custom evaluation with fixed parsing
    cyclener.s2e_model.eval()
    
    predicted_sequences = []
    eval_batch_size = 8
    
    print(f"Evaluating on {len(test_sentences)} samples...")
    
    # Process in smaller batches
    from tqdm import tqdm
    for i in tqdm(range(0, len(test_sentences), eval_batch_size), desc="Validation"):
        batch_sentences = test_sentences[i:i+eval_batch_size]
        torch.cuda.empty_cache()
        batch_predictions = cyclener.s2e_forward(batch_sentences)
        predicted_sequences.extend(batch_predictions)
    
    # Debug first few samples with enhanced parsing
    debug_samples = 5
    print(f"\n=== Enhanced Debugging First {debug_samples} Samples ===")
    
    for i in range(min(debug_samples, len(predicted_sequences))):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*60}")
        
        print(f"Sentence: {test_sentences[i][:200]}...")
        print(f"\nPredicted raw: '{predicted_sequences[i]}'")
        print(f"True raw: '{test_entity_sequences[i]}'")
        
        # Use enhanced parsing for predictions
        try:
            pred_entities = processor.parse_model_output(predicted_sequences[i])
            print(f"\nPredicted entities (robust parser): {pred_entities}")
        except Exception as e:
            print(f"Error parsing predicted entities: {e}")
            pred_entities = []
        
        # Use standard parsing for ground truth
        try:
            true_entities = processor.parse_entity_sequence(test_entity_sequences[i])
            print(f"True entities (standard parser): {true_entities}")
        except Exception as e:
            print(f"Error parsing true entities: {e}")
            true_entities = []
        
        # Calculate sample-level metrics
        if pred_entities or true_entities:
            pred_set = set(pred_entities)
            true_set = set(true_entities)
            overlap = pred_set & true_set
            
            print(f"\nEntity Analysis:")
            print(f"  Overlapping entities: {overlap}")
            print(f"  Predicted only: {pred_set - true_set}")
            print(f"  Missed entities: {true_set - pred_set}")
            
            recall = len(overlap) / len(true_set) if true_set else 0.0
            precision = len(overlap) / len(pred_set) if pred_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            print(f"\nSample Metrics:")
            print(f"  Recall: {recall:.3f} ({len(overlap)}/{len(true_set)})")
            print(f"  Precision: {precision:.3f} ({len(overlap)}/{len(pred_set)})")
            print(f"  F1: {f1:.3f}")
        
        # Format analysis
        pred_has_sep = " <sep> " in predicted_sequences[i]
        true_has_sep = " <sep> " in test_entity_sequences[i]
        print(f"\nFormat Analysis:")
        print(f"  Predicted has <sep>: {pred_has_sep}")
        print(f"  True has <sep>: {true_has_sep}")
        print(f"  Format mismatch: {pred_has_sep != true_has_sep}")
    
    # Calculate metrics using the robust parser for predictions
    print(f"\n{'='*60}")
    print("CALCULATING FINAL METRICS")
    print(f"{'='*60}")
    
    exact_matches = 0
    partial_matches = 0
    format_matches = 0
    parsing_errors = 0
    
    all_predicted_entities = set()
    all_true_entities = set()
    correct_entities = set()
    
    # Track entity type performance
    entity_type_stats = {}
    
    for idx, (pred_seq, true_seq) in enumerate(zip(predicted_sequences, test_entity_sequences)):
        try:
            # Use robust parser for predictions, standard parser for ground truth
            pred_entities = processor.parse_model_output(pred_seq)
            true_entities = processor.parse_entity_sequence(true_seq)
            
            # Convert to sets for comparison
            pred_set = set(pred_entities)
            true_set = set(true_entities)
            
            # Exact match check
            if pred_set == true_set:
                exact_matches += 1
            
            # Partial match check
            if pred_set & true_set:
                partial_matches += 1
            
            # Format match check (both have or don't have <sep>)
            pred_has_sep = " <sep> " in pred_seq
            true_has_sep = " <sep> " in true_seq
            if pred_has_sep == true_has_sep:
                format_matches += 1
            
            # Entity-level metrics
            all_predicted_entities.update(pred_set)
            all_true_entities.update(true_set)
            correct_entities.update(pred_set & true_set)
            
            # Track entity type performance
            for entity, etype in true_entities:
                if etype not in entity_type_stats:
                    entity_type_stats[etype] = {'total': 0, 'found': 0, 'correct': 0}
                entity_type_stats[etype]['total'] += 1
                
                # Check if entity was found (regardless of correctness)
                entity_found = any(e[0] == entity for e in pred_entities)
                if entity_found:
                    entity_type_stats[etype]['found'] += 1
                
                # Check if entity was found with correct type
                if (entity, etype) in pred_set:
                    entity_type_stats[etype]['correct'] += 1
            
        except Exception as e:
            parsing_errors += 1
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Calculate final metrics
    total_samples = len(test_sentences)
    precision = len(correct_entities) / len(all_predicted_entities) if all_predicted_entities else 0.0
    recall = len(correct_entities) / len(all_true_entities) if all_true_entities else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_accuracy = exact_matches / total_samples if total_samples else 0.0
    partial_accuracy = partial_matches / total_samples if total_samples else 0.0
    format_accuracy = format_matches / total_samples if total_samples else 0.0
    
    test_metrics = {
        'total_samples': total_samples,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'format_matches': format_matches,
        'parsing_errors': parsing_errors,
        'exact_accuracy': exact_accuracy,
        'partial_accuracy': partial_accuracy,
        'format_accuracy': format_accuracy,
        'entity_precision': precision,
        'entity_recall': recall,
        'entity_f1': f1,
        'correct_entities': len(correct_entities),
        'predicted_entities': len(all_predicted_entities),
        'true_entities': len(all_true_entities)
    }
    
    # Print comprehensive results
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST RESULTS")
    print(f"{'='*60}")
    
    print(f"\n OVERALL PERFORMANCE:")
    print(f"  Total test samples: {test_metrics['total_samples']:,}")
    print(f"  Parsing errors: {test_metrics['parsing_errors']} ({test_metrics['parsing_errors']/total_samples*100:.1f}%)")
    
    print(f"\n ACCURACY METRICS:")
    print(f"  Exact matches: {test_metrics['exact_matches']:,} ({test_metrics['exact_accuracy']*100:.1f}%)")
    print(f"  Partial matches: {test_metrics['partial_matches']:,} ({test_metrics['partial_accuracy']*100:.1f}%)")
    print(f"  Format matches: {test_metrics['format_matches']:,} ({test_metrics['format_accuracy']*100:.1f}%)")
    
    print(f"\n ENTITY-LEVEL METRICS:")
    print(f"  Entity Precision: {test_metrics['entity_precision']:.4f}")
    print(f"  Entity Recall: {test_metrics['entity_recall']:.4f}")
    print(f"  Entity F1-Score: {test_metrics['entity_f1']:.4f}")
    
    print(f"\n📈 ENTITY COUNTS:")
    print(f"  Correct entities: {test_metrics['correct_entities']:,}")
    print(f"  Predicted entities: {test_metrics['predicted_entities']:,}")
    print(f"  True entities: {test_metrics['true_entities']:,}")
    
    # Entity type breakdown
    if entity_type_stats:
        print(f"\n ENTITY TYPE BREAKDOWN:")
        print(f"{'Type':<12} {'Total':<8} {'Found':<8} {'Correct':<8} {'Recall':<8} {'Precision':<10}")
        print("-" * 65)
        
        for etype, stats in sorted(entity_type_stats.items()):
            total = stats['total']
            found = stats['found']
            correct = stats['correct']
            
            recall = correct / total if total > 0 else 0.0
            precision = correct / found if found > 0 else 0.0
            
            print(f"{etype:<12} {total:<8} {found:<8} {correct:<8} {recall:<8.3f} {precision:<10.3f}")
    
    # Format analysis
    format_issues = total_samples - format_matches
    print(f"\n FORMAT ANALYSIS:")
    print(f"  Samples with format issues: {format_issues} ({format_issues/total_samples*100:.1f}%)")
    print(f"  Expected format: '<entity> <sep> <type> <sep> ...'")
    print(f"  Model produces: '<entity> <type> <entity> <type> ...'")
    
    # Improvement suggestions
    improvement_score = test_metrics['entity_f1']
    print(f"\n💡 PERFORMANCE ASSESSMENT:")
    if improvement_score < 0.3:
        print("  POOR - Major improvements needed")
        print("     Suggestions: Check data quality, adjust training parameters, consider different architecture")
    elif improvement_score < 0.6:
        print("   FAIR - Moderate improvements needed") 
        print("     Suggestions: Fine-tune parsing, add regularization, increase training data")
    elif improvement_score < 0.8:
        print("   GOOD - Minor improvements possible")
        print("     Suggestions: Optimize hyperparameters, post-processing rules")
    else:
        print("   EXCELLENT - Production ready")
    
    return test_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained CycleNER model with fixed parsing')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='Path to checkpoint directory')
    parser.add_argument('--data_path', type=str, 
                       default="solution-synthesis_dataset_2021-8-5.json",
                       help='Path to JSON data file')
    parser.add_argument('--model_name', type=str, default='t5-base', 
                       help='T5 model variant')
    
    args = parser.parse_args()
    
    evaluate_only(args.checkpoint_path, args.data_path, args.model_name)