import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

st.set_page_config(page_title="Llama 3.3 Demo", page_icon="🦙", layout="wide")

st.title("🦙 Llama 3.3 Text Generation")
st.markdown("Generate text using Meta's Llama 3.3 model")

# Check if we're running on Streamlit Cloud
is_streamlit_cloud = os.environ.get('STREAMLIT_CLOUD', False)

if is_streamlit_cloud:
    st.warning("""
    ### ⚠️ Important Note
    This is running on Streamlit Cloud which cannot host the Llama 3.3 model due to size and licensing restrictions.
    
    To use this UI with Llama 3.3:
    1. Clone the repository: `git clone https://github.com/RuthlessGod/llama-3-3-ui.git`
    2. Install dependencies: `pip install -r requirements.txt`
    3. Download Llama 3.3 model (requires access): `python download_model.py --token YOUR_HUGGING_FACE_TOKEN`
    4. Run locally: `streamlit run llama_ui.py`
    
    [Request Llama 3.3 Access](https://huggingface.co/meta-llama/Meta-Llama-3.3-8B)
    """)
    st.stop()

with st.sidebar:
    st.header("Model Settings")
    model_path = st.text_input("Model Path", value="", help="Path to the downloaded Llama 3.3 model")
    use_4bit = st.checkbox("Use 4-bit Quantization", value=True, help="Enable for lower memory usage")
    
    st.header("Generation Settings")
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1, 
                          help="Higher = more creative, Lower = more deterministic")
    max_length = st.slider("Max Length", min_value=64, max_value=2048, value=512, step=64,
                         help="Maximum length of generated text")
    top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.95, step=0.05,
                    help="Nucleus sampling parameter")

@st.cache_resource
def load_model(model_path, use_4bit=False):
    """Load and cache the Llama 3.3 model and tokenizer"""
    if use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=500, temperature=0.7, top_p=0.95):
    """Generate text based on the given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with st.spinner("Generating text..."):
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Input area
prompt = st.text_area("Enter your prompt:", height=150, 
                   value="Explain quantum computing in simple terms")

col1, col2 = st.columns([1, 5])
with col1:
    generate_button = st.button("Generate", type="primary")

with col2:
    if not model_path:
        st.warning("Please enter the path to your Llama 3.3 model in the sidebar.")

# Output area
output_container = st.container()

if generate_button and model_path:
    try:
        model, tokenizer = load_model(model_path, use_4bit)
        generated_text = generate_text(
            model, 
            tokenizer, 
            prompt,
            max_length,
            temperature,
            top_p
        )
        
        with output_container:
            st.markdown("### Generated Text")
            st.markdown(generated_text)
            st.download_button(
                label="Download Result",
                data=generated_text,
                file_name="llama_output.txt",
                mime="text/plain"
            )
    except Exception as e:
        st.error(f"Error loading model or generating text: {str(e)}")
        st.info("Try using 4-bit quantization if you're running out of memory.")