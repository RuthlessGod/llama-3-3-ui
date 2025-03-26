# Llama 3.3 UI

A Streamlit-based user interface for Meta's Llama 3.3 language model.

## Features

- Easy-to-use web interface
- Support for 4-bit quantization for efficient memory usage
- Configurable generation parameters
- Text generation with adjustable settings
- Download generated text as files
- Compatible with both 8B and 70B parameter models

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Access to Meta's Llama 3.3 model

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/RuthlessGod/llama-3-3-ui.git
   cd llama-3-3-ui
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Llama 3.3 model:
   ```bash
   python download_model.py --token YOUR_HUGGING_FACE_TOKEN
   ```

   You'll need to:
   1. Create a Hugging Face account: https://huggingface.co/join
   2. Request access to the model: https://huggingface.co/meta-llama/Meta-Llama-3.3-8B
   3. Create an access token: https://huggingface.co/settings/tokens

## Usage

Run the Streamlit interface:
```bash
streamlit run llama_ui.py
```

Or on Windows, double-click `run_llama_ui.bat`.

## Configuration

In the web interface:
1. Enter the path to your downloaded model
2. Enable 4-bit quantization if needed (recommended for <16GB VRAM)
3. Adjust generation parameters:
   - Temperature (creativity vs. determinism)
   - Maximum length
   - Top-p sampling

## Memory Requirements

- 8B parameter model:
  - Full precision: ~16GB VRAM
  - 4-bit quantization: ~4GB VRAM
- 70B parameter model:
  - Full precision: ~140GB VRAM
  - 4-bit quantization: ~35GB VRAM

## Troubleshooting

1. "CUDA out of memory":
   - Enable 4-bit quantization
   - Reduce max_length
   - Close other GPU applications
   - Use a smaller model

2. Model download issues:
   - Check Hugging Face account access
   - Verify token permissions
   - Ensure stable internet connection

## License

This project is licensed under the MIT License. Note that the Llama 3.3 model has its own license from Meta AI.