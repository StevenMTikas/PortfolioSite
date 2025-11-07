"""Test script to check if .env file is being loaded correctly"""
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get the API key
api_key = os.getenv('OPENAI_API_KEY')

print("=" * 50)
print("Testing .env file loading")
print("=" * 50)

if api_key:
    print(f"✓ API Key loaded successfully")
    print(f"  Length: {len(api_key)}")
    print(f"  Starts with 'sk-': {api_key.startswith('sk-')}")
    print(f"  First 10 chars: {api_key[:10]}...")
    print(f"  Last 10 chars: ...{api_key[-10:]}")
    
    # Check for line breaks or whitespace
    if '\n' in api_key or '\r' in api_key:
        print("  ⚠ WARNING: API key contains line breaks!")
        print(f"  Key with line breaks removed: {api_key.replace(chr(10), '').replace(chr(13), '')}")
    if api_key.strip() != api_key:
        print("  ⚠ WARNING: API key has leading/trailing whitespace!")
        print(f"  Key trimmed: '{api_key.strip()}'")
else:
    print("✗ API Key NOT loaded")
    print("  Check if .env file exists and contains OPENAI_API_KEY")

print("=" * 50)

