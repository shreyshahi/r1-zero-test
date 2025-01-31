# train_grpo.py
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import json
import hashlib
import os
from pathlib import Path
from multiprocessing import Process
from queue import Queue
import multiprocessing as mp
import boto3
from datetime import datetime
from vllm import SamplingParams

SYSTEM_PROMPT = """
Respond only in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

If you do not follow the format, you will be penalized.
Put the final answer in the <answer> tag. Do not use \boxed{} or any other formatting.
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'Use the format described in the system prompt to answer the following question: What is the largest single-digit prime number?'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
                answer="7"
            )},
            {'role': 'user', 'content': f"Use the format described in the system prompt to answer the following question: {x['question']}"}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

train_dataset = get_gsm8k_questions()
test_dataset = get_gsm8k_questions(split="test")

# Add these near the top of the file with other constants
S3_BUCKET_NAME = "gsm8k-grpo-training-traces"  # Replace with your bucket name
S3_PREFIX = f"responses/{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # Creates unique prefix for each run

def file_writer_process(queue):
    """Separate process that handles writing responses to S3"""
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    
    while True:
        data = queue.get()
        if data == "DONE":  # Sentinel value to stop the process
            break
            
        _, current_step, question_id, data_to_write = data
        # Create S3 key (path in bucket)
        s3_key = f"{S3_PREFIX}/step_{current_step}_{question_id}.json"
        
        # Convert data to JSON string
        json_str = json.dumps(data_to_write, indent=2)
        
        # Upload to S3
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=json_str
            )
        except Exception as e:
            print(f"Error uploading to S3: {e}")

# Create a global queue and process
response_queue = mp.Queue()
writer_process = Process(target=file_writer_process, args=(response_queue,))
writer_process.start()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>$"
    responses = [completion[0]["content"].strip() for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [1.0 if match else 0.0 for match in matches]

# Add cleanup function to be called at end of training
def cleanup_writer():
    response_queue.put("DONE")
    writer_process.join()


def evaluate_test_set(trainer, test_dataset, current_step):
    """Use vLLM for efficient batch evaluation"""
    sampling_params = SamplingParams(
        max_tokens=786,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        skip_special_tokens=True
    )

    vllm_engine = trainer.llm
    
    prompts = []
    question_ids = []
    answers = []
    for item in test_dataset:
        prompt = tokenizer.apply_chat_template(item['prompt'], tokenize=False)
        prompts.append(prompt)
        question_ids.append(hashlib.md5(item['prompt'][-1]['content'].encode()).hexdigest())
        answers.append(item['answer'])
    
    # Use the trainer's vLLM instance
    outputs = vllm_engine.generate(prompts, sampling_params)
    
    # Track correct predictions
    correct_count = 0
    total_count = len(outputs)
    
    # Process outputs
    for output, question_id, prompt, answer in zip(outputs, question_ids, prompts, answers):
        response = output.outputs[0].text
        extracted = extract_xml_answer(response)
        
        # Check if prediction matches answer
        if extracted == answer:
            correct_count += 1
        
        # Queue data for writing
        data = {
            "id": question_id,
            "step": current_step,
            "question": prompt,
            "answer": answer,
            "response": response,
            "extracted": extracted,
            "split": "test"
        }
        response_queue.put((None, current_step, question_id, data))
    
    # Calculate accuracy
    accuracy = correct_count / total_count
    
    if "validation_accuracy" not in trainer._metrics:
        trainer._metrics["validation_accuracy"] = []
    
    trainer._metrics["validation_accuracy"].append(accuracy)

class TestEvalCallback(TrainerCallback):
    def __init__(self, trainer, test_dataset):
        self.trainer = trainer
        self.test_dataset = test_dataset

    def on_train_begin(self, args, state, control, **kwargs):
        # Run evaluation at step 0 before training starts
        evaluate_test_set(self.trainer, self.test_dataset, 0)
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            evaluate_test_set(self.trainer, self.test_dataset, state.global_step)

model_name = "llama1b"  # Path to local model folder

output_dir = "outputs/Llama-1B-GRPO"
run_name = "Llama-1B-GRPO-gsm8k"
    
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=25,
    save_steps=10000,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    use_vllm=True,  # Enable vLLM for faster generation
    vllm_device="cuda:1",
    vllm_gpu_memory_utilization=0.4,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
    trust_remote_code=True,  # Added for loading local model
).to("cuda")

# Set max_model_len after initialization
model.max_model_len = 4096 

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True  # Added for loading local model
)
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
)

callback = TestEvalCallback(
    trainer=trainer,
    test_dataset=test_dataset
)
trainer.add_callback(callback)

try:
    trainer.train()
finally:
    cleanup_writer()  # Ensure we clean up the writer process
