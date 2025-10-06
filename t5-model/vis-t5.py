import json
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from train import CycleNER, MaterialsDataProcessor

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class FixedMaterialsDataProcessor(MaterialsDataProcessor):
    """Extended processor with robust parsing for model outputs"""

    def parse_entity_sequence(self, sequence: str) -> list:
        if not sequence or not sequence.strip():
            return []
        
        sequence = sequence.strip()
        
        if " <sep> " not in sequence:
            return []
        
        parts = sequence.split(" <sep> ")
        entities = []
        
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                entity = parts[i].strip()
                entity_type = parts[i + 1].strip()
                if entity and entity_type:
                    entities.append((entity, entity_type))
        
        return entities
    
    def parse_model_output(self, sequence: str) -> list:
        if not sequence or not sequence.strip():
            return []
        
        sequence = sequence.strip()
        entities = []
        
        entity_types = {'PRECURSOR', 'TARGET', 'SOLVENT', 'TEMPERATURE', 'TIME', 'OPERATION', 'QUANTITY'}
        
        if " <sep> " in sequence:
            return self.parse_entity_sequence(sequence)
        
        tokens = sequence.split()
        i = 0
        
        while i < len(tokens):
            entity_found = False
            
            for entity_len in range(1, 5):
                if i < len(tokens) - entity_len and tokens[i + entity_len] in entity_types:
                    entity = " ".join(tokens[i:i + entity_len])
                    entity_type = tokens[i + entity_len]
                    entities.append((entity, entity_type))
                    i += entity_len + 1
                    entity_found = True
                    break
            
            if not entity_found:
                i += 1
        
        if not entities:
            pattern = r'([A-Za-z0-9\(\)\.\-]+(?:\s+[A-Za-z0-9\(\)\.\-°]+)*)\s+(PRECURSOR|TARGET|SOLVENT|TEMPERATURE|TIME|OPERATION|QUANTITY)'
            matches = re.findall(pattern, sequence)
            entities = [(match[0].strip(), match[1].strip()) for match in matches]
        
        seen = set()
        unique_entities = []
        for entity, etype in entities:
            key = (entity, etype)
            if key not in seen and entity and etype:
                seen.add(key)
                unique_entities.append((entity, etype))
        
        return unique_entities

def plot_results(test_metrics, entity_type_stats, output_prefix="t5"):
    """Create and save comprehensive plots"""
    
    # 1. Overall Performance Metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
    
    # Accuracy metrics
    ax1 = axes[0, 0]
    accuracy_types = ['Exact', 'Partial', 'Format']
    accuracy_values = [
        test_metrics['exact_accuracy'] * 100,
        test_metrics['partial_accuracy'] * 100,
        test_metrics['format_accuracy'] * 100
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax1.bar(accuracy_types, accuracy_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Metrics', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Entity-level metrics
    ax2 = axes[0, 1]
    entity_metrics = ['Precision', 'Recall', 'F1-Score']
    entity_values = [
        test_metrics['entity_precision'] * 100,
        test_metrics['entity_recall'] * 100,
        test_metrics['entity_f1'] * 100
    ]
    colors2 = ['#9b59b6', '#f39c12', '#1abc9c']
    bars2 = ax2.bar(entity_metrics, entity_values, color=colors2, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Score (%)', fontsize=12)
    ax2.set_title('Entity-Level Metrics', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Entity counts
    ax3 = axes[1, 0]
    entity_counts = ['Correct', 'Predicted', 'True']
    count_values = [
        test_metrics['correct_entities'],
        test_metrics['predicted_entities'],
        test_metrics['true_entities']
    ]
    colors3 = ['#27ae60', '#e67e22', '#34495e']
    bars3 = ax3.bar(entity_counts, count_values, color=colors3, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Entity Counts', fontsize=13, fontweight='bold')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # Match distribution
    ax4 = axes[1, 1]
    match_labels = ['Exact\nMatches', 'Partial\nMatches', 'No\nMatch']
    match_values = [
        test_metrics['exact_matches'],
        test_metrics['partial_matches'] - test_metrics['exact_matches'],
        test_metrics['total_samples'] - test_metrics['partial_matches']
    ]
    colors4 = ['#2ecc71', '#f39c12', '#e74c3c']
    wedges, texts, autotexts = ax4.pie(match_values, labels=match_labels, colors=colors4,
                                         autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax4.set_title('Match Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_overall_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_overall_metrics.png")
    plt.close()
    
    # 2. Entity Type Performance
    if entity_type_stats:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Entity Type Performance', fontsize=16, fontweight='bold')
        
        entity_types = sorted(entity_type_stats.keys())
        recalls = [entity_type_stats[et]['correct'] / entity_type_stats[et]['total'] 
                   if entity_type_stats[et]['total'] > 0 else 0 for et in entity_types]
        precisions = [entity_type_stats[et]['correct'] / entity_type_stats[et]['found'] 
                      if entity_type_stats[et]['found'] > 0 else 0 for et in entity_types]
        
        # Recall and Precision comparison
        ax1 = axes[0]
        x = np.arange(len(entity_types))
        width = 0.35
        bars1 = ax1.bar(x - width/2, [r*100 for r in recalls], width, label='Recall', 
                        color='#3498db', alpha=0.7, edgecolor='black')
        bars2 = ax1.bar(x + width/2, [p*100 for p in precisions], width, label='Precision',
                        color='#e74c3c', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Entity Type', fontsize=12)
        ax1.set_ylabel('Score (%)', fontsize=12)
        ax1.set_title('Recall vs Precision by Entity Type', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(entity_types, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Entity counts by type
        ax2 = axes[1]
        totals = [entity_type_stats[et]['total'] for et in entity_types]
        corrects = [entity_type_stats[et]['correct'] for et in entity_types]
        
        bars1 = ax2.bar(x - width/2, totals, width, label='Total', 
                        color='#95a5a6', alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x + width/2, corrects, width, label='Correct',
                        color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Entity Type', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Total vs Correct Entities by Type', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(entity_types, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_entity_type_performance.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_prefix}_entity_type_performance.png")
        plt.close()
        
        # 3. F1 Score heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        f1_scores = []
        for et in entity_types:
            r = recalls[entity_types.index(et)]
            p = precisions[entity_types.index(et)]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_scores.append(f1 * 100)
        
        colors_map = plt.cm.RdYlGn(np.array(f1_scores) / 100)
        bars = ax.barh(entity_types, f1_scores, color=colors_map, edgecolor='black', alpha=0.8)
        ax.set_xlabel('F1 Score (%)', fontsize=12)
        ax.set_ylabel('Entity Type', fontsize=12)
        ax.set_title('F1 Score by Entity Type', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            ax.text(score + 2, i, f'{score:.1f}%', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_f1_scores.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_prefix}_f1_scores.png")
        plt.close()

def evaluate_only(checkpoint_path, data_path, model_name='t5-base'):
    """Load trained model and evaluate on test data with fixed parsing"""
    
    print(f"Loading data from {data_path}")
    with open(data_path, 'r') as f:
        json_data = json.load(f)
    
    processor = FixedMaterialsDataProcessor()
    sentences, entity_sequences = processor.process_json_data(json_data)
    
    split_idx = int(0.8 * len(sentences))
    test_sentences = sentences[split_idx:]
    test_entity_sequences = entity_sequences[split_idx:]
    
    print(f"Test data: {len(test_sentences)} sentences")
    
    cyclener = CycleNER(model_name=model_name)
    
    print(f"Loading model from {checkpoint_path}")
    try:
        cyclener.load_checkpoint(checkpoint_path, model_name)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
    
    print("Running evaluation on full test set...")
    
    cyclener.s2e_model.eval()
    
    predicted_sequences = []
    eval_batch_size = 8
    
    print(f"Evaluating on {len(test_sentences)} samples...")
    
    from tqdm import tqdm
    for i in tqdm(range(0, len(test_sentences), eval_batch_size), desc="Validation"):
        batch_sentences = test_sentences[i:i+eval_batch_size]
        torch.cuda.empty_cache()
        batch_predictions = cyclener.s2e_forward(batch_sentences)
        predicted_sequences.extend(batch_predictions)
    
    debug_samples = 5
    print(f"\n=== Enhanced Debugging First {debug_samples} Samples ===")
    
    for i in range(min(debug_samples, len(predicted_sequences))):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*60}")
        
        print(f"Sentence: {test_sentences[i][:200]}...")
        print(f"\nPredicted raw: '{predicted_sequences[i]}'")
        print(f"True raw: '{test_entity_sequences[i]}'")
        
        try:
            pred_entities = processor.parse_model_output(predicted_sequences[i])
            print(f"\nPredicted entities (robust parser): {pred_entities}")
        except Exception as e:
            print(f"Error parsing predicted entities: {e}")
            pred_entities = []
        
        try:
            true_entities = processor.parse_entity_sequence(test_entity_sequences[i])
            print(f"True entities (standard parser): {true_entities}")
        except Exception as e:
            print(f"Error parsing true entities: {e}")
            true_entities = []
        
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
        
        pred_has_sep = " <sep> " in predicted_sequences[i]
        true_has_sep = " <sep> " in test_entity_sequences[i]
        print(f"\nFormat Analysis:")
        print(f"  Predicted has <sep>: {pred_has_sep}")
        print(f"  True has <sep>: {true_has_sep}")
        print(f"  Format mismatch: {pred_has_sep != true_has_sep}")
    
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
    
    entity_type_stats = {}
    
    for idx, (pred_seq, true_seq) in enumerate(zip(predicted_sequences, test_entity_sequences)):
        try:
            pred_entities = processor.parse_model_output(pred_seq)
            true_entities = processor.parse_entity_sequence(true_seq)
            
            pred_set = set(pred_entities)
            true_set = set(true_entities)
            
            if pred_set == true_set:
                exact_matches += 1
            
            if pred_set & true_set:
                partial_matches += 1
            
            pred_has_sep = " <sep> " in pred_seq
            true_has_sep = " <sep> " in true_seq
            if pred_has_sep == true_has_sep:
                format_matches += 1
            
            all_predicted_entities.update(pred_set)
            all_true_entities.update(true_set)
            correct_entities.update(pred_set & true_set)
            
            for entity, etype in true_entities:
                if etype not in entity_type_stats:
                    entity_type_stats[etype] = {'total': 0, 'found': 0, 'correct': 0}
                entity_type_stats[etype]['total'] += 1
                
                entity_found = any(e[0] == entity for e in pred_entities)
                if entity_found:
                    entity_type_stats[etype]['found'] += 1
                
                if (entity, etype) in pred_set:
                    entity_type_stats[etype]['correct'] += 1
            
        except Exception as e:
            parsing_errors += 1
            print(f"Error processing sample {idx}: {e}")
            continue
    
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
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST RESULTS")
    print(f"{'='*60}")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total test samples: {test_metrics['total_samples']:,}")
    print(f"  Parsing errors: {test_metrics['parsing_errors']} ({test_metrics['parsing_errors']/total_samples*100:.1f}%)")
    
    print(f"\nACCURACY METRICS:")
    print(f"  Exact matches: {test_metrics['exact_matches']:,} ({test_metrics['exact_accuracy']*100:.1f}%)")
    print(f"  Partial matches: {test_metrics['partial_matches']:,} ({test_metrics['partial_accuracy']*100:.1f}%)")
    print(f"  Format matches: {test_metrics['format_matches']:,} ({test_metrics['format_accuracy']*100:.1f}%)")
    
    print(f"\nENTITY-LEVEL METRICS:")
    print(f"  Entity Precision: {test_metrics['entity_precision']:.4f}")
    print(f"  Entity Recall: {test_metrics['entity_recall']:.4f}")
    print(f"  Entity F1-Score: {test_metrics['entity_f1']:.4f}")
    
    print(f"\nENTITY COUNTS:")
    print(f"  Correct entities: {test_metrics['correct_entities']:,}")
    print(f"  Predicted entities: {test_metrics['predicted_entities']:,}")
    print(f"  True entities: {test_metrics['true_entities']:,}")
    
    if entity_type_stats:
        print(f"\nENTITY TYPE BREAKDOWN:")
        print(f"{'Type':<12} {'Total':<8} {'Found':<8} {'Correct':<8} {'Recall':<8} {'Precision':<10}")
        print("-" * 65)
        
        for etype, stats in sorted(entity_type_stats.items()):
            total = stats['total']
            found = stats['found']
            correct = stats['correct']
            
            recall_val = correct / total if total > 0 else 0.0
            precision_val = correct / found if found > 0 else 0.0
            
            print(f"{etype:<12} {total:<8} {found:<8} {correct:<8} {recall_val:<8.3f} {precision_val:<10.3f}")
    
    format_issues = total_samples - format_matches
    print(f"\nFORMAT ANALYSIS:")
    print(f"  Samples with format issues: {format_issues} ({format_issues/total_samples*100:.1f}%)")
    
    improvement_score = test_metrics['entity_f1']
    print(f"\nPERFORMANCE ASSESSMENT:")
    if improvement_score < 0.3:
        print("  POOR - Major improvements needed")
    elif improvement_score < 0.6:
        print("  FAIR - Moderate improvements needed") 
    elif improvement_score < 0.8:
        print("  GOOD - Minor improvements possible")
    else:
        print("  EXCELLENT - Production ready")
    
    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    plot_results(test_metrics, entity_type_stats, output_prefix="t5")
    
    return test_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained CycleNER model with plots')
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints3/final_checkpoint/", 
                       help='Path to checkpoint directory')
    parser.add_argument('--data_path', type=str, 
                       default="./solution-synthesis_dataset_2021-8-5.json",
                       help='Path to JSON data file')
    parser.add_argument('--model_name', type=str, default='t5-base', 
                       help='T5 model variant')
    
    args = parser.parse_args()
    
    evaluate_only(args.checkpoint_path, args.data_path, args.model_name)