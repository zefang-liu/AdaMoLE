"""
Testing LLMs on Benchmarks
"""
import argparse
import json
import os
import re

import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    GenerationConfig,
)
from transformers.pipelines.pt_utils import KeyDataset

from data import get_formatted_datasets
from src import PeftConfig, PeftModelForCausalLM

transformers.set_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_choices(examples):
    """
    Predict choices
    """
    prompts = examples['text']
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :]
    choices = [chr(ord('A') + i) for i in range(max(examples['num_choices']))]
    choice_ids = [tokenizer.encode(choice, add_special_tokens=False)[-1] for choice in choices]

    predicted_ids = torch.argmax(logits[:, choice_ids], dim=-1)
    predictions = [choices[predicted_id] for predicted_id in predicted_ids.cpu().numpy()]
    examples['prediction'] = predictions

    return examples


if __name__ == '__main__':
    # Add arguments
    parser = argparse.ArgumentParser(
        description='Fine-tuning LLMs on training data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument(
        '--model_path', type=str, default='outputs/llama-2-7b-hf-adamole-the8-commonsense-qa',
        help='huggingface model id or local model path')
    parser.add_argument(
        '--data_path', type=str, default='tau/commonsense_qa',
        help='huggingface data id or local data path')
    parser.add_argument(
        '--max_new_tokens', type=int, default=16,
        help='maximum number of new tokens')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='batch size in the pipeline')
    parser.add_argument(
        '--logits', default=False, action='store_true',
        help='checking choice logits instead of generated texts')

    # Parse arguments
    args = parser.parse_args()
    model_path = args.model_path
    data_path = args.data_path
    model_name = os.path.basename(model_path).lower()
    data_name = os.path.basename(data_path).lower()
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size
    if data_name in ['openbookqa', 'ai2_arc']:
        split = 'test'
    else:
        split = 'validation'

    # Load and format datasets
    formatted_datasets = get_formatted_datasets(data_path=data_path, prompt_only=True)

    # Load the configuration and model
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        # torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModelForCausalLM.from_pretrained(model=base_model, model_id=model_path)
    model.to(device)
    print(f'Model loaded from {model_path}')
    print(f'Model: {model}')

    if not args.logits:
        # Build the pipeline
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        # Get the model responses
        responses = []
        for response in tqdm(
            pipeline(
                KeyDataset(formatted_datasets[split], 'text'),
                generation_config=generation_config,
                return_full_text=False,
                batch_size=batch_size,
            ),
            total=len(formatted_datasets[split]),
        ):
            responses.append(response[0]['generated_text'])

        # Print one response
        print(f'Response example:\n{responses[0]}')

        # Get the results
        df = formatted_datasets[split].to_pandas()
        df['response'] = responses
        df['prediction'] = df['response'].str.extract(pat=r'\b([A-Z])\b')[0]

    else:
        # Get predictions
        dataset_with_predictions = formatted_datasets[split].map(
            predict_choices, batched=True, batch_size=batch_size)
        df = dataset_with_predictions.to_pandas()

    # Save the results
    result_path = os.path.join(model_path, f'{split}_results.csv')
    df.to_csv(result_path, index=False)
    print(f'Results saved to {result_path}')

    # Compute evaluation metrics
    metrics = {}
    for _data_name in df['data_name'].unique():
        df_set = df[df['data_name'] == _data_name]
        accuracy = pd.Series(df_set['answer'] == df_set['prediction']).mean()
        print(f'Accuracy of {_data_name}: {accuracy:.2%}')
        metrics['accuracy_' + re.sub(r'\W', '_', _data_name)] = accuracy

    # Save evaluation metrics
    metric_path = os.path.join(model_path, f'{split}_metrics.json')
    with open(metric_path, 'w') as file:
        json.dump(metrics, file)
    print(f'Metrics saved to {metric_path}')
