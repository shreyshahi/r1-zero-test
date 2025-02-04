import os
import json

def check_file_correctness(file_path):
    """Check if a file meets the correctness criteria"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Get the answer from the data
    correct_answer = data.get('answer')
    if not correct_answer:
        return False
        
    # Sort steps numerically
    steps = sorted(data['steps'].keys(), key=lambda x: int(x))
    
    # Check early steps (< 500) are wrong 90% of the time
    early_steps = [step for step in steps if int(step) < 500]
    early_wrong_count = sum(
        1 for step in early_steps
        if data['steps'][step]['extracted'] != correct_answer
    )
    early_wrong_ratio = early_wrong_count / len(early_steps) if early_steps else 0
    
    # Check late steps (> 1800) are correct 90% of the time
    late_steps = [step for step in steps if int(step) > 1400 and int(step) <= 1500]
    late_correct_count = sum(
        1 for step in late_steps
        if data['steps'][step]['extracted'] == correct_answer
    )
    late_correct_ratio = late_correct_count / len(late_steps) if late_steps else 0
    
    return early_wrong_ratio >= 0.75 and late_correct_ratio >= 0.95

def generate_manifest():
    # Source and destination directories
    source_dir = "../validation_data"
    dest_dir = "public/data"
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Process each JSON file
    processed_files = []
    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            source_path = os.path.join(source_dir, filename)
            
            # Check if file meets criteria
            if check_file_correctness(source_path):
                # Copy file to destination, keeping only first 1500 steps
                with open(source_path, 'r') as src_file:
                    data = json.load(src_file)
                    # Filter steps to keep only first 1500
                    steps = data['steps']
                    filtered_steps = {k: v for k, v in steps.items() if int(k) <= 1500}
                    data['steps'] = filtered_steps
                    
                    dest_path = os.path.join(dest_dir, filename)
                    with open(dest_path, 'w') as dest_file:
                        json.dump(data, dest_file, indent=2)
                    processed_files.append(filename)
    
    print(f"Processed {len(processed_files)} files that met criteria")
    
    # Write manifest
    manifest_path = os.path.join(dest_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(processed_files, f, indent=2)
    print(f"Manifest file created at {manifest_path}")

if __name__ == "__main__":
    generate_manifest()