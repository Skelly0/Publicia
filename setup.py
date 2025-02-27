#!/usr/bin/env python3
"""
Setup script for Publicia Discord Bot
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    print("Setting up Publicia Discord Bot...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Install requirements
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Error: Failed to install dependencies")
        sys.exit(1)
    
    # Create required directories
    dirs = ["documents", "lorebooks", "conversations", "images", "user_preferences", "avatars"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")
    
    # Check for .env file
    if not os.path.exists(".env"):
        print("\nCreating sample .env file...")
        with open(".env", "w") as f:
            f.write("# Discord Bot Token from Discord Developer Portal\n")
            f.write("DISCORD_BOT_TOKEN=your_token_here\n\n")
            f.write("# OpenRouter API Key\n")
            f.write("OPENROUTER_API_KEY=your_openrouter_key_here\n\n")
            f.write("# LLM Model to use (e.g., deepseek/deepseek-r1)\n")
            f.write("LLM_MODEL=deepseek/deepseek-r1\n\n")
            f.write("# Classifier Model for query analysis\n")
            f.write("CLASSIFIER_MODEL=google/gemini-2.0-flash-001\n\n")
            f.write("# Number of search results to return\n")
            f.write("TOP_K=10\n\n")
            f.write("# API timeout in seconds\n")
            f.write("API_TIMEOUT=150\n\n")
            f.write("# Maximum number of retries\n")
            f.write("MAX_RETRIES=10\n")
        print("Created sample .env file. Please edit it with your actual credentials.")
    
    print("\nSetup complete! To run the bot:")
    print("1. Edit the .env file with your actual tokens and keys")
    print("2. Run the bot with: python PubliciaV8.py")

if __name__ == "__main__":
    main()
