import os
import argparse
from huggingface_hub import snapshot_download

def download_model(model_name, output_dir=None):
    """Download a model from Hugging Face"""
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "models", model_name.split("/")[-1])
    
    print(f"Downloading model {model_name} to {output_dir}")
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            token=os.environ.get("HUGGINGFACE_TOKEN"),  # Set this env var if using private models
            local_dir_use_symlinks=False  # Important for Windows compatibility
        )
        print(f"Model successfully downloaded to {output_dir}")
        print(f"You can now use it with: --model_path \"{output_dir}\"")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print("\nIf you're trying to access Meta-Llama-3.3 models, you need to:")
        print("1. Create a Hugging Face account: https://huggingface.co/join")
        print("2. Request access to the model: https://huggingface.co/meta-llama/Meta-Llama-3.3-8B")
        print("3. Create an access token: https://huggingface.co/settings/tokens")
        print("4. Set the HUGGINGFACE_TOKEN environment variable:")
        print("   - On Windows: set HUGGINGFACE_TOKEN=your_token_here")
        print("   - Or pass it directly: python download_model.py --token your_token_here")

def main():
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.3-8B", 
                        help="Model name to download (default: meta-llama/Meta-Llama-3.3-8B)")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save the model (default: ./models/[model_name])")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token for accessing private models")
    
    args = parser.parse_args()
    
    if args.token:
        os.environ["HUGGINGFACE_TOKEN"] = args.token
    
    download_model(args.model, args.output_dir)

if __name__ == "__main__":
    main()