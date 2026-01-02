import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Checking imports...")
try:
    from src.config import ModelConfig
    print("✅ src.config imported")
    from src.data import load_movie_dataset
    print("✅ src.data imported")
    from src.model import load_model, add_lora_adapters
    print("✅ src.model imported")
    from src.train import train_model
    print("✅ src.train imported")
    from src.inference import generate_response
    print("✅ src.inference imported")
    print("All imports successful. Structure OK.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
