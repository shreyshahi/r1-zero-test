import aioboto3
import asyncio
import os
import json
from collections import defaultdict
from typing import Dict, List, Set
import argparse
from botocore.exceptions import NoCredentialsError
import aiofiles

# S3 bucket configuration
S3_BUCKET_NAME = "gsm8k-grpo-training-traces"
MAX_CONCURRENT_DOWNLOADS = 50  # Adjust based on your needs

def get_existing_files(raw_data_dir: str) -> Set[str]:
    """Get set of existing files in the raw data directory"""
    if not os.path.exists(raw_data_dir):
        return set()
    return {f for f in os.listdir(raw_data_dir) if f.endswith('.json')}

async def download_file(session, bucket: str, key: str, local_path: str):
    """Download a single file from S3 asynchronously"""
    async with session.client('s3') as s3_client:
        try:
            response = await s3_client.get_object(Bucket=bucket, Key=key)
            async with aiofiles.open(local_path, 'wb') as f:
                async with response['Body'] as stream:
                    await f.write(await stream.read())
        except Exception as e:
            print(f"Error downloading {key}: {e}")

async def download_s3_data(timestamp: str, raw_data_dir: str, force: bool = False) -> None:
    """Download data from S3 for the specified timestamp using concurrent downloads"""
    try:
        session = aioboto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
    except NoCredentialsError:
        print("Error: AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        exit(1)
    
    # Create directory if it doesn't exist
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Get existing files
    existing_files = get_existing_files(raw_data_dir) if not force else set()
    if existing_files:
        print(f"Found {len(existing_files)} existing files")
    
    # List objects in the bucket with the specified prefix
    prefix = f"responses/{timestamp}"
    
    try:
        async with session.client('s3') as s3_client:
            paginator = s3_client.get_paginator('list_objects_v2')
            
            found_files = False
            download_tasks = []
            s3_files = set()
            
            # First, list all files in S3
            async for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                    
                found_files = True
                
                # Create download tasks for each new file
                for obj in page['Contents']:
                    relative_path = obj['Key']
                    filename = os.path.basename(relative_path)
                    s3_files.add(filename)
                    
                    # Skip if file already exists
                    if filename in existing_files:
                        continue
                        
                    local_path = os.path.join(raw_data_dir, filename)
                    task = download_file(session, S3_BUCKET_NAME, relative_path, local_path)
                    download_tasks.append(task)
            
            if not found_files:
                print(f"No files found for timestamp: {timestamp}")
                exit(1)
            
            # Report on file status
            total_s3_files = len(s3_files)
            new_files = len(download_tasks)
            skipped_files = total_s3_files - new_files
            
            print(f"Found {total_s3_files} files in S3")
            print(f"Skipping {skipped_files} already downloaded files")
            
            # Download new files concurrently in batches
            if new_files > 0:
                print(f"Downloading {new_files} new files...")
                
                for i in range(0, new_files, MAX_CONCURRENT_DOWNLOADS):
                    batch = download_tasks[i:i + MAX_CONCURRENT_DOWNLOADS]
                    await asyncio.gather(*batch)
                    print(f"Progress: {min(i + MAX_CONCURRENT_DOWNLOADS, new_files)}/{new_files} files")
                
                print(f"Downloaded {new_files} new files to {raw_data_dir}")
            else:
                print("All files already downloaded")
            
            # Check for any files that exist locally but not in S3
            local_only = existing_files - s3_files
            if local_only:
                print(f"Warning: Found {len(local_only)} files in local directory that don't exist in S3")
                print("These files might be from a different timestamp or could be corrupted")
            
    except Exception as e:
        print(f"Error accessing S3: {e}")
        exit(1)

def process_validation_data(raw_data_dir: str, processed_data_dir: str) -> None:
    """Process the raw validation data and combine by question ID"""
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Dictionary to store data by ID
    data_by_id: Dict[str, Dict] = defaultdict(lambda: {
        "id": "",
        "question": "",
        "answer": "",
        "steps": {}
    })
    
    # Dictionary to store accuracy by step
    accuracy_by_step: Dict[int, List] = defaultdict(lambda: [])
    
    # Process each file in the raw data directory
    file_count = 0
    for filename in os.listdir(raw_data_dir):
        if not filename.endswith('.json'):
            continue
            
        with open(os.path.join(raw_data_dir, filename), 'r') as f:
            data = json.load(f)
        
        # Only process test split data
        if data.get('split') != 'test':
            continue
            
        file_count += 1
        question_id = data['id']
        step = data['step']
        
        # Store basic information
        if not data_by_id[question_id]['id']:
            data_by_id[question_id].update({
                'id': question_id,
                'question': data['question'],
                'answer': data['answer']
            })
        
        # Store step information
        data_by_id[question_id]['steps'][step] = {
            'response': data['response'],
            'extracted': data['extracted']
        }
        
        # Track accuracy
        is_correct = data['extracted'] == data['answer']
        accuracy_by_step[step].append(is_correct)
    
    if file_count == 0:
        print("No test data files found to process")
        return
    
    # Save processed data
    for question_id, data in data_by_id.items():
        output_file = os.path.join(processed_data_dir, f"{question_id}.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Print accuracy by step
    print("\nValidation Accuracy by Step:")
    for step in sorted(accuracy_by_step.keys()):
        accuracies = accuracy_by_step[step]
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"Step {step}: {avg_accuracy:.4f} ({len(accuracies)} samples)")

async def async_main():
    parser = argparse.ArgumentParser(description='Download and process validation data from S3')
    parser.add_argument('--timestamp', type=str, required=True,
                      help='Timestamp of the training run (format: YYYYMMDD_HHMMSS)')
    parser.add_argument('--raw-dir', type=str, default="raw_validation_data",
                      help='Directory for raw downloaded data')
    parser.add_argument('--processed-dir', type=str, default="validation_data",
                      help='Directory for processed data')
    parser.add_argument('--concurrent-downloads', type=int, default=50,
                      help='Maximum number of concurrent downloads')
    parser.add_argument('--force', action='store_true',
                      help='Force download all files, ignoring existing ones')
    
    args = parser.parse_args()
    
    global MAX_CONCURRENT_DOWNLOADS
    MAX_CONCURRENT_DOWNLOADS = args.concurrent_downloads
    
    # Download data asynchronously
    await download_s3_data(args.timestamp, args.raw_dir, args.force)
    
    # Process data (synchronous)
    process_validation_data(args.raw_dir, args.processed_dir)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
