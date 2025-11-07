"""Quick test script to verify chatbot setup"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("Testing Chatbot Setup")
print("=" * 50)

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✓ OPENAI_API_KEY is set (length: {len(api_key)})")
else:
    print("✗ OPENAI_API_KEY is NOT set in .env file")

# Check PDF files
mebot_folder = Path("assets/MeBot")
if mebot_folder.exists():
    print(f"✓ MeBot folder exists: {mebot_folder}")
    pdf_files = list(mebot_folder.glob("*.pdf"))
    if pdf_files:
        print(f"✓ Found {len(pdf_files)} PDF file(s):")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
    else:
        print("✗ No PDF files found in MeBot folder")
else:
    print(f"✗ MeBot folder does NOT exist: {mebot_folder}")

print("=" * 50)

