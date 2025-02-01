import os
import json

def generate_manifest():
    # Path to your validation data directory
    data_dir = "public/data"
    
    # Get all JSON files
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {data_dir}")
    
    # Write the manifest file
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(json_files, f)
    print(f"Manifest file created at {manifest_path}")

if __name__ == "__main__":
    generate_manifest()