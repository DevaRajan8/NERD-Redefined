import streamlit as st
import json
import torch
import pandas as pd
import os
import re
import requests
import time
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.optim as optim
from typing import List, Dict, Tuple, TypedDict
import pdfplumber
from langgraph.graph import StateGraph, END
import graphviz

st.set_page_config(
    page_title="PDF Entity Extraction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


class GraphState(TypedDict):
    """State definition for LangGraph workflow"""
    pdf_text: str
    chunks: List[str]
    raw_entities: List[Dict]
    validated_entities: List[Dict]
    test_sentences: List[str]
    test_sequences: List[str]
    predictions: List[str]
    error: str


class PDFEntityExtractor:
    """Extract entities using LangGraph workflow"""
    
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.groq_api_key = api_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model_name = model_name
        self.entity_types = {'PRECURSOR', 'TARGET', 'SOLVENT', 'TEMPERATURE', 
                            'TIME', 'OPERATION', 'QUANTITY', 'MATERIAL', 'CHEMICAL'}
    
    def call_groq_api(self, messages: list, max_retries: int = 3) -> str:
        """Call Groq API with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.1
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    wait_time = (attempt + 1) * 5
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                if attempt == max_retries - 1:
                    raise Exception(f"Groq API call failed: {str(e)}")
                time.sleep(3)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Groq API call failed: {str(e)}")
                time.sleep(3)
        return ""
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF"""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'- ', '', text)
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def chunk_text(self, state: GraphState) -> GraphState:
        """Node 1: Split text into chunks"""
        text = state['pdf_text']
        chunk_size = 4000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        state['chunks'] = chunks
        st.info(f"Split text into {len(chunks)} chunks")
        return state
    
    def extract_entities(self, state: GraphState) -> GraphState:
        """Node 2: Extract entities from chunks"""
        chunks = state['chunks']
        all_entities = []
        
        system_prompt = """You are an expert materials science entity extraction system. 
Extract entities and return them as a JSON array with this exact format:
[
  {"entity": "titanium isopropoxide", "type": "PRECURSOR"},
  {"entity": "80°C", "type": "TEMPERATURE"}
]

Entity Types:
- PRECURSOR: Starting materials, reactants
- TARGET: Final products being synthesized
- SOLVENT: Liquid mediums
- TEMPERATURE: Temperature values
- TIME: Duration or time periods
- OPERATION: Synthesis methods, processes
- QUANTITY: Amounts with units
- MATERIAL: General materials, equipment
- CHEMICAL: Chemical formulas, compounds

Rules:
1. Return ONLY valid JSON array
2. Extract real substances, values, or processes
3. Be precise and comprehensive
4. NO explanations or markdown"""

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, chunk in enumerate(chunks):
            status_text.text(f"Processing chunk {idx + 1}/{len(chunks)}...")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract all materials science entities:\n\n{chunk}"}
            ]
            
            try:
                response = self.call_groq_api(messages)
                content = response.strip()
                
                # Clean response
                content = re.sub(r'^```json\s*', '', content)
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    entities_list = json.loads(json_match.group())
                    for ent in entities_list:
                        if isinstance(ent, dict) and 'entity' in ent and 'type' in ent:
                            all_entities.append({
                                'entity': ent['entity'].strip(),
                                'type': ent['type'].strip().upper()
                            })
            except Exception as e:
                st.warning(f"Chunk {idx + 1}: {str(e)[:100]}")
            
            progress_bar.progress((idx + 1) / len(chunks))
        
        progress_bar.empty()
        status_text.empty()
        
        state['raw_entities'] = all_entities
        st.success(f"Extracted {len(all_entities)} raw entities")
        return state
    
    def validate_entities(self, state: GraphState) -> GraphState:
        """Node 3: Validate and deduplicate entities"""
        entities = state['raw_entities']
        
        # Validation
        valid_entities = []
        for ent in entities:
            entity_text = ent['entity'].strip()
            entity_type = ent['type'].strip().upper()
            
            if not entity_text or not entity_type:
                continue
            if len(entity_text) > 200 or len(entity_text) < 1:
                continue
            if entity_type not in self.entity_types:
                continue
            
            bad_phrases = ['extract', 'format', 'return', 'example']
            if any(phrase in entity_text.lower() for phrase in bad_phrases):
                continue
            
            valid_entities.append(ent)
        
        # Deduplication
        entity_map = {}
        for ent in valid_entities:
            key = ent['entity'].lower().strip()
            if key in entity_map:
                priority = {'TARGET': 5, 'PRECURSOR': 4, 'CHEMICAL': 3, 'MATERIAL': 2}
                old_priority = priority.get(entity_map[key]['type'], 1)
                new_priority = priority.get(ent['type'], 1)
                if new_priority > old_priority:
                    entity_map[key] = ent
            else:
                entity_map[key] = ent
        
        state['validated_entities'] = list(entity_map.values())
        st.success(f"Validated {len(state['validated_entities'])} unique entities")
        return state
    
    def create_test_data(self, state: GraphState) -> GraphState:
        """Node 4: Create test pairs"""
        text = state['pdf_text']
        entities = state['validated_entities']
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if 30 < len(s.strip()) < 1000]
        
        entity_dict = {e['entity'].lower(): e for e in entities}
        
        test_pairs = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            found_entities = []
            
            for entity_key, ent in entity_dict.items():
                if entity_key in sentence_lower:
                    found_entities.append((ent['entity'], ent['type']))
            
            # Changed from >= 2 to >= 1 to include ALL sentences with entities
            if len(found_entities) >= 1:
                found_entities = list(dict.fromkeys(found_entities))
                entity_parts = []
                for entity_text, entity_type in found_entities:
                    entity_parts.extend([entity_text, entity_type])
                
                entity_sequence = " <sep> ".join(entity_parts)
                test_pairs.append((sentence, entity_sequence))
        
        test_pairs.sort(key=lambda x: x[1].count('<sep>'), reverse=True)
        
        state['test_sentences'] = [p[0] for p in test_pairs]
        state['test_sequences'] = [p[1] for p in test_pairs]
        st.success(f"Created {len(test_pairs)} test pairs (covering all extracted entities)")
        return state
    
    def build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("chunk_text", self.chunk_text)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("validate_entities", self.validate_entities)
        workflow.add_node("create_test_data", self.create_test_data)
        
        workflow.set_entry_point("chunk_text")
        workflow.add_edge("chunk_text", "extract_entities")
        workflow.add_edge("extract_entities", "validate_entities")
        workflow.add_edge("validate_entities", "create_test_data")
        workflow.add_edge("create_test_data", END)
        
        return workflow.compile()
    
    def visualize_graph(self):
        """Create visual representation of the graph"""
        dot = graphviz.Digraph(comment='Entity Extraction Workflow')
        dot.attr(rankdir='LR')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
        dot.node('START', 'START', shape='circle', fillcolor='lightgreen')
        dot.node('chunk', 'Chunk Text')
        dot.node('extract', 'Extract Entities')
        dot.node('validate', 'Validate & Deduplicate')
        dot.node('test', 'Create Test Data')
        dot.node('END', 'END', shape='circle', fillcolor='lightcoral')
        
        dot.edge('START', 'chunk')
        dot.edge('chunk', 'extract')
        dot.edge('extract', 'validate')
        dot.edge('validate', 'test')
        dot.edge('test', 'END')
        
        return dot


class MaterialsDataProcessor:
    """Process materials science data"""
    
    def parse_model_output(self, sequence: str) -> List[Tuple[str, str]]:
        """Parse model output"""
        if not sequence or not sequence.strip():
            return []
        
        entities = []
        entity_types = {'PRECURSOR', 'TARGET', 'SOLVENT', 'TEMPERATURE', 'TIME', 
                       'OPERATION', 'QUANTITY', 'MATERIAL', 'CHEMICAL'}
        
        if " <sep> " in sequence or "<sep>" in sequence:
            sequence = sequence.replace("<sep>", " <sep> ")
            parts = [p.strip() for p in sequence.split(" <sep> ") if p.strip()]
            
            for i in range(0, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    entity = parts[i].strip()
                    entity_type = parts[i + 1].strip().upper()
                    if entity_type in entity_types and entity:
                        entities.append((entity, entity_type))
        
        return list(dict.fromkeys(entities))
    
    def parse_raw_prediction(self, prediction: str) -> List[Tuple[str, str]]:
        """Parse raw unstructured predictions like 'dried OPERATION water SOLVENT'"""
        if not prediction or not prediction.strip():
            return []
        
        entities = []
        entity_types = {'PRECURSOR', 'TARGET', 'SOLVENT', 'TEMPERATURE', 'TIME', 
                       'OPERATION', 'QUANTITY', 'MATERIAL', 'CHEMICAL'}
        
        # Split by spaces and look for type keywords
        tokens = prediction.split()
        i = 0
        
        while i < len(tokens):
            # Check if current token is a type
            if tokens[i].upper() in entity_types:
                # Look back to get the entity
                if i > 0:
                    entity = tokens[i-1]
                    entity_type = tokens[i].upper()
                    entities.append((entity, entity_type))
                i += 1
            else:
                i += 1
        
        return list(dict.fromkeys(entities))


class CycleNER:
    """BART NER Model"""
    
    def __init__(self, model_name="facebook/bart-base", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        special_tokens = {"additional_special_tokens": ["<sep>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        
        self.s2e_model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        self.s2e_model.resize_token_embeddings(len(self.tokenizer))
        self.s2e_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.s2e_optimizer = optim.Adam(self.s2e_model.parameters(), lr=5e-5)
    
    def encode_batch(self, texts: List[str], max_length: int = 512):
        encoded = self.tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def decode_batch(self, token_ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    def s2e_forward(self, sentences: List[str], batch_size: int = 8) -> List[str]:
        if not sentences:
            return []
        
        all_predictions = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            inputs = self.encode_batch(batch_sentences)
            
            with torch.no_grad():
                outputs = self.s2e_model.generate(
                    **inputs, max_length=256, num_beams=4,
                    temperature=1.0, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            all_predictions.extend(self.decode_batch(outputs))
            torch.cuda.empty_cache()
            
            progress = min((i + batch_size) / len(sentences), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {i + len(batch_sentences)}/{len(sentences)} sentences")
        
        progress_bar.empty()
        status_text.empty()
        
        return all_predictions
    
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.s2e_model.load_state_dict(checkpoint['s2e_model_state'])
        self.s2e_optimizer.load_state_dict(checkpoint['s2e_optimizer_state'])


@st.cache_resource
def load_model(checkpoint_path):
    """Load BART model"""
    try:
        cyclener = CycleNER()
        cyclener.load_checkpoint(checkpoint_path)
        cyclener.s2e_model.eval()
        return cyclener
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def main():
    st.markdown('<h1 class="main-header">PDF Entity Extraction System</h1>', unsafe_allow_html=True)
    st.markdown("Advanced materials science entity extraction using LangGraph")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    groq_model = st.sidebar.selectbox(
        "Select Model",
        ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )
    
    checkpoint_path = st.sidebar.text_input(
        "BART Checkpoint Path",
        value="./bart_checkpoints_2/final_checkpoint/checkpoint.pt"
    )
    
    if st.sidebar.button("Load BART Model"):
        if checkpoint_path and os.path.exists(checkpoint_path):
            with st.spinner("Loading model..."):
                model = load_model(checkpoint_path)
            if model:
                st.sidebar.success("Model loaded successfully")
                st.session_state.model = model
        else:
            st.sidebar.error("Checkpoint not found")
    
    if 'model' in st.session_state:
        st.sidebar.success("Model: Ready")
    else:
        st.sidebar.warning("Model: Not loaded")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Extract Entities", "Test Model", "Predictions", "LangGraph Flow"])
    
    with tab1:
        st.subheader("Step 1: Extract Entities from PDF")
        
        uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
        
        if uploaded_file and groq_api_key:
            if st.button("Extract Entities", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        extractor = PDFEntityExtractor(groq_api_key, groq_model)
                        
                        uploaded_file.seek(0)
                        pdf_text = extractor.extract_text_from_pdf(uploaded_file)
                        
                        if len(pdf_text.strip()) < 100:
                            st.error("PDF contains insufficient text")
                        else:
                            st.success(f"Extracted {len(pdf_text):,} characters")
                            
                            # Run LangGraph workflow
                            graph = extractor.build_graph()
                            initial_state = GraphState(
                                pdf_text=pdf_text, chunks=[], raw_entities=[],
                                validated_entities=[], test_sentences=[],
                                test_sequences=[], predictions=[], error=""
                            )
                            
                            st.info("Running LangGraph workflow...")
                            result = graph.invoke(initial_state)
                            
                            entities = result['validated_entities']
                            st.session_state['pdf_text'] = pdf_text
                            st.session_state['entities'] = entities
                            st.session_state['test_sentences'] = result['test_sentences']
                            st.session_state['test_sequences'] = result['test_sequences']
                            
                            df = pd.DataFrame(entities)
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.dataframe(df, use_container_width=True, height=400)
                            
                            with col2:
                                st.write("**Distribution**")
                                entity_counts = df['type'].value_counts()
                                for etype, count in entity_counts.items():
                                    st.metric(etype, count)
                            
                            st.bar_chart(entity_counts)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                json_str = json.dumps(entities, indent=2)
                                st.download_button("Download JSON", json_str, "entities.json", "application/json")
                            with col2:
                                csv_str = df.to_csv(index=False)
                                st.download_button("Download CSV", csv_str, "entities.csv", "text/csv")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Step 2: Test BART Model")
        
        if 'model' not in st.session_state:
            st.warning("Please load BART model first")
            return
        
        if 'test_sentences' not in st.session_state:
            st.info("Extract entities first")
            return
        
        st.metric("Test Pairs Ready", len(st.session_state['test_sentences']))
        
        if st.button("Run Predictions", type="primary"):
            with st.spinner("Running predictions..."):
                try:
                    model = st.session_state['model']
                    test_sentences = st.session_state['test_sentences']
                    
                    predictions = model.s2e_forward(test_sentences, batch_size=8)
                    st.session_state['predictions'] = predictions
                    
                    st.success(f"Generated {len(predictions)} predictions")
                    st.info("Go to 'Predictions' tab to view results")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab3:
        st.subheader("Step 3: View Predictions")
        
        if 'predictions' not in st.session_state:
            st.info("Run predictions first")
            return
        
        processor = MaterialsDataProcessor()
        test_sentences = st.session_state['test_sentences']
        predictions = st.session_state['predictions']
        
        st.metric("Total Predictions", len(predictions))
        
        # Parse predictions to extract entities
        all_predicted_entities = []
        for i, (sentence, prediction) in enumerate(zip(test_sentences, predictions)):
            # Try to parse structured format first
            pred_entities = processor.parse_model_output(prediction)
            
            if pred_entities:
                for entity, entity_type in pred_entities:
                    all_predicted_entities.append({
                        'Entity': entity,
                        'Type': entity_type
                    })
            else:
                # Parse raw unstructured predictions
                raw_entities = processor.parse_raw_prediction(prediction)
                for entity, entity_type in raw_entities:
                    all_predicted_entities.append({
                        'Entity': entity,
                        'Type': entity_type
                    })
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for ent in all_predicted_entities:
            key = (ent['Entity'].lower(), ent['Type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
        
        if not unique_entities:
            st.warning("No entities were parsed from predictions.")
            return
        
        results_df = pd.DataFrame(unique_entities)
        
        st.write(f"**Total Unique Entities Predicted: {len(results_df)}**")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            search_entity = st.text_input("Search Entity")
        with col2:
            unique_types = sorted(results_df['Type'].unique().tolist()) if len(results_df) > 0 else []
            filter_type = st.multiselect("Filter by Type", options=unique_types)
        
        filtered_df = results_df.copy()
        if search_entity:
            filtered_df = filtered_df[filtered_df['Entity'].str.contains(search_entity, case=False, na=False)]
        if filter_type:
            filtered_df = filtered_df[filtered_df['Type'].isin(filter_type)]
        
        st.dataframe(filtered_df, use_container_width=True, height=500)
        
        # Download simple Entity, Type CSV
        col1, col2 = st.columns(2)
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button("Download Predictions CSV", csv, "predicted_entities.csv", "text/csv", use_container_width=True)
        
        with col2:
            # Also offer extracted entities from step 1
            if 'entities' in st.session_state:
                extracted_df = pd.DataFrame(st.session_state['entities'])[['entity', 'type']]
                extracted_df.columns = ['Entity', 'Type']
                extracted_csv = extracted_df.to_csv(index=False)
                st.download_button(
                    f"Download All Extracted Entities ({len(extracted_df)})", 
                    extracted_csv, 
                    "extracted_entities.csv", 
                    "text/csv",
                    use_container_width=True
                )
        
        # Summary by type
        st.subheader("Summary by Entity Type")
        type_counts = results_df['Type'].value_counts()
        st.bar_chart(type_counts)

if __name__ == "__main__":
    main()