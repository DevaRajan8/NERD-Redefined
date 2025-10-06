import streamlit as st
import json
import torch
import pandas as pd
import os
from transformers import AutoTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration
import torch.optim as optim
from typing import List, Dict, Tuple
import tempfile

# Set page config
st.set_page_config(
    page_title="Materials Science NER",
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
    def __init__(self, model_name="facebook/bart-base", model_type="bart", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_type = model_type.lower()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens
        special_tokens = {"additional_special_tokens": ["<sep>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Initialize model based on type
        if self.model_type == "bart":
            self.s2e_model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        elif self.model_type == "t5":
            self.s2e_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'bart' or 't5'")
        
        # Resize embeddings for new tokens
        self.s2e_model.resize_token_embeddings(len(self.tokenizer))
        
        # Configure model
        self.s2e_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.s2e_model.config.eos_token_id = self.tokenizer.eos_token_id
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            self.s2e_model.config.bos_token_id = self.tokenizer.bos_token_id
            self.s2e_model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        
        # Initialize optimizer (needed for loading checkpoint)
        self.s2e_optimizer = optim.Adam(self.s2e_model.parameters(), lr=5e-5)
    
    def encode_batch(self, texts: List[str], max_length: int = 512):
        """Encode batch of texts"""
        # Add task prefix for T5
        if self.model_type == "t5":
            texts = [f"extract entities: {text}" for text in texts]
        
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
                    num_beams=4,
                    temperature=0.7,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            batch_predictions = self.decode_batch(outputs)
            all_predictions.extend(batch_predictions)
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_predictions
    
    def load_checkpoint(self, checkpoint_path: str, model_name: str = None):
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
def load_model(checkpoint_path, model_name, model_type):
    """Load the trained model (cached)"""
    try:
        cyclener = CycleNER(model_name=model_name, model_type=model_type)
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
    
    # Model type selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        options=["BART", "T5"],
        help="Select the model architecture"
    )
    
    # Model name input
    if model_type == "BART":
        default_model = "facebook/bart-base"
    else:
        default_model = "t5-small"
    
    model_name = st.sidebar.text_input(
        "Model Name",
        value=default_model,
        help="HuggingFace model name (e.g., facebook/bart-base or t5-small)"
    )
    
    # Checkpoint path
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path", 
        value="checkpoint",
        help="Path to your trained model checkpoint file"
    )
    
    # Load model button
    if st.sidebar.button("Load Model"):
        if checkpoint_path and os.path.exists(checkpoint_path):
            with st.spinner("Loading model..."):
                model = load_model(checkpoint_path, model_name, model_type.lower())
            if model:
                st.sidebar.success("✅ Model loaded successfully!")
                st.session_state.model = model
                st.session_state.model_type = model_type
            else:
                st.sidebar.error("❌ Failed to load model")
        else:
            st.sidebar.error("❌ Checkpoint file not found")
    
    # Display model info if loaded
    if 'model' in st.session_state:
        st.sidebar.info(f"**Loaded Model:** {st.session_state.model_type}\n\n**Architecture:** {model_name}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Input Data")
        
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
            placeholder='Paste your JSON data here...\n\nExample format:\n[\n  {\n    "paragraph_string": "Your text here...",\n    ...\n  }\n]'
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
                    
                    # Parse entities - Create detailed results
                    all_results = []
                    
                    for i, (sentence, pred_seq) in enumerate(zip(sentences, predicted_sequences)):
                        entities = processor.parse_model_output(pred_seq)
                        
                        if entities:
                            # Add each entity as a separate row
                            for entity, entity_type in entities:
                                all_results.append({
                                    'Entity': entity,
                                    'Entity_Type': entity_type
                                })
                        else:
                            # Add a row even if no entities found
                            all_results.append({
                                'Entity': '',
                                'Entity_Type': ''
                            })
                    
                    if all_results:
                        # Create DataFrame
                        df = pd.DataFrame(all_results)
                        
                        # Count only non-empty entities
                        non_empty_entities = df[df['Entity'] != '']
                        
                        # Display results
                        st.success(f"✅ Extracted {len(non_empty_entities)} entities from {len(sentences)} sentences")
                        
                        # Show statistics
                        if len(non_empty_entities) > 0:
                            entity_counts = non_empty_entities['Entity_Type'].value_counts()
                            st.markdown("**📊 Entity Type Distribution:**")
                            col_stat1, col_stat2 = st.columns(2)
                            
                            with col_stat1:
                                for entity_type, count in list(entity_counts.items())[:4]:
                                    st.metric(entity_type, count)
                            
                            with col_stat2:
                                for entity_type, count in list(entity_counts.items())[4:]:
                                    st.metric(entity_type, count)
                        
                        # Display dataframe
                        st.subheader("📋 Extracted Entities")
                        st.dataframe(df, use_container_width=True, height=400)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv,
                            file_name="extracted_entities.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
                        # Show sample predictions
                        st.subheader("🔍 Sample Predictions")
                        for i in range(min(3, len(sentences))):
                            with st.expander(f"Sample {i+1}"):
                                st.write(f"**Sentence:** {sentences[i][:300]}{'...' if len(sentences[i]) > 300 else ''}")
                                st.write(f"**Raw Prediction:** `{predicted_sequences[i]}`")
                                
                                sample_entities = processor.parse_model_output(predicted_sequences[i])
                                if sample_entities:
                                    st.write("**Parsed Entities:**")
                                    for entity, etype in sample_entities:
                                        st.write(f"- **{entity}** → `{etype}`")
                                else:
                                    st.write("*No entities extracted*")
                    
                    else:
                        st.warning("⚠️ No entities were extracted from the input data")
                
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()