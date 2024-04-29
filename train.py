"""
Fine-Tuning LLMs on Tasks
"""
import argparse
import os
import re

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from data import get_formatted_datasets
from src import (
    TaskType,
    LoraConfig,
    MoleConfig,
    AdaMoleConfig,
    PeftTrainer,
    PeftModelForCausalLM,
)

transformers.set_seed(0)

if __name__ == '__main__':
    # Add arguments
    parser = argparse.ArgumentParser(
        description='Fine-tuning LLMs on training data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument(
        '--model_path', type=str, default='meta-llama/Llama-2-7b-hf',
        help='huggingface model id or local model path')
    parser.add_argument(
        '--data_path', type=str, default='tau/commonsense_qa',
        help='huggingface data id or local data path')
    parser.add_argument(
        '--peft_type', type=str, default='lora', choices=['lora', 'mole', 'adamole'],
        help='peft model type to be fine-tuned')
    parser.add_argument(
        '--lora_rank', type=int, default=32,
        help='lora rank if the peft type is lora or total lora rank if moe')
    parser.add_argument(
        '--target_modules', type=str, default=['q_proj', 'v_proj'], nargs='+',
        help='target modules in lora layers')
    parser.add_argument(
        '--num_experts', type=int, default=1,
        help='number of experts in each moe layer')
    parser.add_argument(
        '--top_k', type=int, default=None,
        help='top-k experts in moe (only one of top_k or threshold can be used)')
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='threshold for expert gating in moe (only one of top_k or threshold can be used)')
    parser.add_argument(
        '--max_length', type=int, default=256,
        help='maximum number of tokens')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='batch size in the trainer')
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help='gradient accumulation steps')
    parser.add_argument(
        '--num_train_epochs', type=int, default=1,
        help='number of training epochs')
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='learning rate for training')
    parser.add_argument(
        '--lr_scheduler_type', type=str, default="constant_with_warmup",
        help='learning rate scheduler type')
    parser.add_argument(
        '--warmup_steps', type=int, default=200,
        help='number of warmup steps for training')
    parser.add_argument(
        '--weight_decay', type=float, default=0.0,
        help='weight decay')
    parser.add_argument(
        '--aux_loss_coeff', type=float, default=None,
        help='auxiliary loss coefficient for moe')

    # Parse arguments
    args = parser.parse_args()
    print(f'Arguments: {args}')
    model_path = args.model_path
    data_path = args.data_path
    model_name = os.path.basename(model_path).lower()
    data_name = os.path.basename(data_path).lower()
    peft_type = args.peft_type
    num_experts = args.num_experts
    max_length = args.max_length
    lora_rank = args.lora_rank if peft_type == 'lora' else args.lora_rank // num_experts
    lora_alpha = 16
    lora_dropout = 0.05
    peft_type_name = peft_type
    if args.top_k is not None:
        peft_type_name += f'-top{args.top_k}'
    if args.threshold is not None:
        threshold_name = int(1 / args.threshold)
        peft_type_name += f'-the{threshold_name}'
    output_dir = os.path.join('outputs', re.sub(r'[^0-9a-zA-Z]', '-', f'{model_name}-{peft_type_name}-{data_name}'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and format datasets
    formatted_datasets = get_formatted_datasets(data_path=data_path, prompt_only=False)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        # add_bos_token=True,
        add_eos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize datasets
    tokenize_text = lambda examples: tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        # padding=True,
        # return_tensors="pt",
    )
    tokenized_datasets = formatted_datasets.map(
        tokenize_text,
        batched=True,
        remove_columns=formatted_datasets["train"].column_names,
    )
    print(f'Tokenized datasets: {tokenized_datasets}')

    # Set the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt")

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # torch_dtype=torch.bfloat16,
        # device_map="auto",
    )
    print(f'Base model loaded from {model_path}')
    print(f'Base model: {base_model}')

    # Get the PEFT model
    if peft_type == 'lora':
        peft_config = LoraConfig(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=args.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
    elif peft_type == 'mole':
        peft_config = MoleConfig(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=args.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            num_experts=num_experts,
            top_k=args.top_k,
            threshold=args.threshold,
        )
    elif peft_type == 'adamole':
        peft_config = AdaMoleConfig(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=args.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            num_experts=num_experts,
            max_threshold=args.threshold,
        )
    else:
        raise KeyError(f'Unsupported PEFT type: {peft_type}')

    model = PeftModelForCausalLM(base_model, peft_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print(f'Model: {model}')

    # Set the trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        group_by_length=True,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        # save_steps=1000,
        optim="adamw_torch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        # fp16=True,
        seed=0,
        data_seed=0,
        report_to=["tensorboard"],
    )
    trainer = PeftTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        aux_loss_coeff=args.aux_loss_coeff,
    )
    with open(os.path.join(output_dir, 'training_args.json'), 'w') as output_file:
        output_file.write(training_args.to_json_string())

    # Train the model
    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True

    # Save the model
    trainer.save_model()
    trainer.save_state()
