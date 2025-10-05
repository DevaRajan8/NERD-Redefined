import streamlit as st
import json
import torch
import pandas as pd
import os
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.optim as optim
from typing import List, Dict, Tuple
import tempfile

# Set page config
st.set_page_config(
    page_title="Materials Science NER",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MaterialsDataProcessor:
    """Process materials science JSON data for NER"""
    
    def __init__(self):
        # Define material entity types
        self.entity_types = {
            'PRECURSOR': 'precursor',
            'TARGET': 'target', 
            'SOLVENT': 'solvent',
            'TEMPERATURE': 'temperature',
            'TIME': 'time',
            'OPERATION': 'operation',
            'QUANTITY': 'quantity'
        }
    
    def parse_model_output(self, sequence: str) -> List[Tuple[str, str]]:
        """Robust parser that handles multiple output formats from the model"""
        if not sequence or not sequence.strip():
            return []
        
        sequence = sequence.strip()
        entities = []
        
        # Define known entity types
        entity_types = {'PRECURSOR', 'TARGET', 'SOLVENT', 'TEMPERATURE', 'TIME', 'OPERATION', 'QUANTITY'}
        
        # Method 1: Standard <sep> format
        if " <sep> " in sequence:
            parts = sequence.split(" <sep> ")
            for i in range(0, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    entity = parts[i].strip()
                    entity_type = parts[i + 1].strip()
                    if entity and entity_type:
                        entities.append((entity, entity_type))
            return entities
        
        # Method 2: Space-separated format (what model actually produces)
        tokens = sequence.split()
        i = 0
        
        while i < len(tokens):
            entity_found = False
            
            # Try different entity lengths (1-4 words) followed by type
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
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity, etype in entities:
            key = (entity, etype)
            if key not in seen and entity and etype:
                seen.add(key)
                unique_entities.append((entity, etype))
        
        return unique_entities

class CycleNER:
    def __init__(self, model_name="facebook/bart-base", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens
        special_tokens = {"additional_special_tokens": ["<sep>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Initialize models
        self.s2e_model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # Resize embeddings for new tokens
        self.s2e_model.resize_token_embeddings(len(self.tokenizer))
        
        # Configure model
        self.s2e_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.s2e_model.config.eos_token_id = self.tokenizer.eos_token_id
        self.s2e_model.config.bos_token_id = self.tokenizer.bos_token_id
        self.s2e_model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        
        # Initialize optimizer (needed for loading checkpoint)
        self.s2e_optimizer = optim.Adam(self.s2e_model.parameters(), lr=5e-5)
    
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
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
            inputs = self.encode_batch(batch_sentences)
            
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
    
    def load_checkpoint(self, checkpoint_path: str, model_name: str = "facebook/bart-base"):
        """Load checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.s2e_model.load_state_dict(checkpoint['s2e_model_state'])
        self.s2e_optimizer.load_state_dict(checkpoint['s2e_optimizer_state'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")

@st.cache_resource
def load_model(checkpoint_path):
    """Load the trained model (cached)"""
    try:
        cyclener = CycleNER()
        cyclener.load_checkpoint(checkpoint_path)
        cyclener.s2e_model.eval()
        return cyclener
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def extract_sentences_from_json(json_data):
    """Extract sentences from JSON data"""
    sentences = []
    for record in json_data:
        paragraph = record.get('paragraph_string', '').strip()
        if paragraph:
            sentences.append(paragraph)
    return sentences

def main():
    st.title("🧪 Materials Science Named Entity Recognition")
    st.markdown("Extract entities from materials science literature using trained CycleNER model")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model loading
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path", 
        value="C:\\Users\\rdeva\\Downloads\\sem5\\NERD-Redefined\\finalcheckpoint\\checkpoint.pt",
        help="Path to your trained model checkpoint file"
    )
    
    # Load model button
    if st.sidebar.button("Load Model"):
        if checkpoint_path and os.path.exists(checkpoint_path):
            with st.spinner("Loading model..."):
                model = load_model(checkpoint_path)
            if model:
                st.sidebar.success("✅ Model loaded successfully!")
                st.session_state.model = model
            else:
                st.sidebar.error("❌ Failed to load model")
        else:
            st.sidebar.error("❌ Checkpoint file not found")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Input Data")
        
        # Show expected JSON format
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload JSON file", 
            type="json",
            help="Upload a JSON file containing materials science data"
        )
        
        # Text input option
        st.markdown("**Or paste JSON data:**")
        json_text = st.text_area(
            "JSON Data",
            height=200,
            placeholder="Paste your JSON data here..."
        )
    
    with col2:
        st.subheader("🔬 Entity Extraction")
        
        # Check if model is loaded
        if 'model' not in st.session_state:
            st.warning("⚠️ Please load the model first using the sidebar")
            return
        
        model = st.session_state.model
        processor = MaterialsDataProcessor()
        
        # Process input
        json_data = None
        
        if uploaded_file is not None:
            try:
                json_data = json.load(uploaded_file)
                st.success(f"✅ File uploaded: {len(json_data)} records")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        
        elif json_text.strip():
            try:
                json_data = json.loads(json_text)
                if isinstance(json_data, dict):
                    json_data = [json_data]  # Convert single record to list
                st.success(f"✅ JSON parsed: {len(json_data)} records")
            except Exception as e:
                st.error(f"❌ Error parsing JSON: {str(e)}")
        
        # Process data if available
        if json_data and st.button("🚀 Extract Entities", type="primary"):
            with st.spinner("Extracting entities..."):
                try:
                    # Extract sentences
                    sentences = extract_sentences_from_json(json_data)
                    
                    if not sentences:
                        st.error("❌ No valid sentences found in the data")
                        return
                    
                    st.info(f"📄 Processing {len(sentences)} sentences...")
                    
                    # Get predictions
                    predicted_sequences = model.s2e_forward(sentences)
                    
                    # Parse entities
                    all_results = []
                    
                    for i, (sentence, pred_seq) in enumerate(zip(sentences, predicted_sequences)):
                        entities = processor.parse_model_output(pred_seq)
                        
                        # Add to results
                        for entity, entity_type in entities:
                            all_results.append({
                                'Record_ID': i + 1,
                                'Sentence': sentence[:100] + "..." if len(sentence) > 100 else sentence,
                                'Entity': entity,
                                'Entity_Type': entity_type,
                                'Raw_Prediction': pred_seq
                            })
                    
                    if all_results:
                        # Create DataFrame
                        df = pd.DataFrame(all_results)
                        
                        # Display results
                        st.success(f"Extracted {len(all_results)} entities from {len(sentences)} sentences")
                        
                        # Show statistics
                        entity_counts = df['Entity_Type'].value_counts()
                        st.markdown("**Entity Type Distribution:**")
                        for entity_type, count in entity_counts.items():
                            st.write(f"- **{entity_type}**: {count} entities")
                        
                        # Display dataframe
                        st.subheader("Extracted Entities")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="extracted_entities.csv",
                            mime="text/csv"
                        )
                        
                        # Show sample predictions
                        st.subheader("Sample Predictions")
                        for i in range(min(3, len(sentences))):
                            with st.expander(f"Sample {i+1}"):
                                st.write(f"**Sentence:** {sentences[i][:200]}...")
                                st.write(f"**Raw Prediction:** {predicted_sequences[i]}")
                                
                                sample_entities = processor.parse_model_output(predicted_sequences[i])
                                if sample_entities:
                                    st.write("**Parsed Entities:**")
                                    for entity, etype in sample_entities:
                                        st.write(f"- **{entity}** → {etype}")
                                else:
                                    st.write("No entities extracted")
                    
                    else:
                        st.warning("⚠️ No entities were extracted from the input data")
                
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **About:** This app uses a trained CycleNER model for Named Entity Recognition in materials science literature.
        
        **Entity Types:** PRECURSOR, TARGET, SOLVENT, TEMPERATURE, TIME, OPERATION, QUANTITY
        
        **Model Architecture:** BART-based sequence-to-sequence model with cycle-consistency training
        """
    )

if __name__ == "__main__":
    main()