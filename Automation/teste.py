import os
from pathlib import Path

home = Path.home()
config_file = home / ".onshape_client_config"

print("--- Onshape Credential Diagnostic ---")

# 1. Check for the hidden file
if config_file.exists():
    print(f"[FOUND] Config file exists at: {config_file}")
    print("-> The script is likely using keys from this file (Old Account).")
    print("-> ACTION: Delete this file to use Environment Variables.")

    # Optional: Read the file to see the key (partial)
    try:
        with open(config_file, 'r') as f:
            content = f.read()
            if "access_key" in content:
                print("   (File contains an access key)")
    except:
        pass
else:
    print("[CLEAN] No config file found in user folder.")

print("\n2. Check Environment Variables")
access_key = os.environ.get('ONSHAPE_API_ACCESS_KEY')
if access_key:
    print(f"[FOUND] Environment Variable Set: {access_key[:5]}... (Partial)")
    print("-> If the file above is deleted, the script will use this key.")
else:
    print("[MISSING] No 'ONSHAPE_API_ACCESS_KEY' found in environment.")

print("-" * 30)