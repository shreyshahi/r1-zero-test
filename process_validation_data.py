import aioboto3
import asyncio
import os
import json
from collections import defaultdict
from typing import Dict, List
import argparse
from botocore.exceptions import NoCredentialsError
import aiofiles

# S3 bucket configuration
S3_BUCKET_NAME = "gsm8k-grpo-training-traces"
MAX_CONCURRENT_DOWNLOADS = 50  # Adjust based on your needs

async def download_file(session, bucket: str, key: str, local_path: str):
    """Download a single file from S3 asynchronously"""
    if os.path.exists(local_path):
        return
        
    async with session.client('s3') as s3_client:
        try:
            response = await s3_client.get_object(Bucket=bucket, Key=key)
            async with aiofiles.open(local_path, 'wb') as f:
                async with response['Body'] as stream:
                    await f.write(await stream.read())
        except Exception as e:
            print(f"Error downloading {key}: {e}")

async def download_s3_data(timestamp: str, raw_data_dir: str) -> None:
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
    
    # List objects in the bucket with the specified prefix
    prefix = f"responses/{timestamp}"
    
    try:
        async with session.client('s3') as s3_client:
            paginator = s3_client.get_paginator('list_objects_v2')
            
            found_files = False
            download_tasks = []
            
            async for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                    
                found_files = True
                
                # Create download tasks for each file
                for obj in page['Contents']:
                    relative_path = obj['Key']
                    local_path = os.path.join(raw_data_dir, os.path.basename(relative_path))
                    
                    # Skip if file already exists
                    if os.path.exists(local_path):
                        continue
                        
                    task = download_file(session, S3_BUCKET_NAME, relative_path, local_path)
                    download_tasks.append(task)
            
            if not found_files:
                print(f"No files found for timestamp: {timestamp}")
                exit(1)
            
            # Download files concurrently in batches
            total_files = len(download_tasks)
            if total_files > 0:
                print(f"Downloading {total_files} files...")
                
                for i in range(0, total_files, MAX_CONCURRENT_DOWNLOADS):
                    batch = download_tasks[i:i + MAX_CONCURRENT_DOWNLOADS]
                    await asyncio.gather(*batch)
                    print(f"Progress: {min(i + MAX_CONCURRENT_DOWNLOADS, total_files)}/{total_files} files")
            
            print(f"Downloaded data to {raw_data_dir}")
            
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
    
    args = parser.parse_args()
    
    global MAX_CONCURRENT_DOWNLOADS
    MAX_CONCURRENT_DOWNLOADS = args.concurrent_downloads
    
    # Download data asynchronously
    await download_s3_data(args.timestamp, args.raw_dir)
    
    # Process data (synchronous)
    process_validation_data(args.raw_dir, args.processed_dir)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
