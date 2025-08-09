from LAMAR.modeling_nucESM2 import EsmForMaskedLM
from transformers import AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset
import argparse


def main(
        tokenizer_path, 
        model_max_length, 
        model_name, 
        token_dropout, 
        positional_embedding_type, 
        hidden_size, 
        intermediate_size, 
        num_attention_heads, 
        num_hidden_layers, 
        data_for_pretrain_path,  
        flash_attention, 
        disable_tqdm, 
        batch_size, 
        peak_lr, 
        warmup_ratio, 
        max_steps,  
        grad_clipping_norm, 
        accum_steps, 
        output_dir, 
        save_steps, 
        logging_steps, 
        fp16, 
        resume_training, 
        data_collator_patch
    ):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    # Config
    config = AutoConfig.from_pretrained(
        model_name, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id, 
        token_dropout=token_dropout, positional_embedding_type=positional_embedding_type, 
        hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers
    )
    # Training data
    def group_texts(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length)

    train_set = load_dataset("text", data_files=data_for_pretrain_path, streaming=True)
    data_for_pretrain = train_set.map(group_texts, remove_columns=["text"])
    # Data Collator
    if data_collator_patch:
        from LAMAR.data_collator_patch import DataCollatorForLanguageModeling_patch
        data_collator = DataCollatorForLanguageModeling_patch(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # Model
    model = EsmForMaskedLM(config)
    if flash_attention:
        from LAMAR.flash_attn_patch import EsmSelfAttentionAddFlashAttnPatch
        for i in range(config.num_hidden_layers):
            model.esm.encoder.layer[i].attention.self = EsmSelfAttentionAddFlashAttnPatch(config, position_embedding_type='rotary')
    # Training arguments
    train_args = TrainingArguments(
        disable_tqdm=disable_tqdm, 
        save_total_limit=100, 
        dataloader_drop_last=True, 
        per_device_train_batch_size=batch_size, 
        learning_rate=peak_lr, 
        weight_decay=0.01, 
        adam_beta1=0.9, 
        adam_beta2=0.98, 
        adam_epsilon=1e-8, 
        warmup_ratio=warmup_ratio, 
        max_steps=max_steps,
        max_grad_norm=grad_clipping_norm, 
        gradient_accumulation_steps=accum_steps, 
        output_dir=output_dir, 
        save_strategy='steps', 
        save_steps=save_steps, 
        logging_steps=logging_steps, 
        fp16=fp16,  
        half_precision_backend='apex',
        fp16_opt_level='O2',
        dispatch_batches=False, 
        ignore_data_skip=True, 
        report_to='none'
    )
    # Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=data_for_pretrain['train'], 
        data_collator=data_collator, 
        tokenizer=tokenizer
    )
    # Training
    trainer.train(resume_from_checkpoint=resume_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretraining LAMAR')
    parser.add_argument('--tokenizer_path', default='tokenizer/single_nucleotide', type=str, help='Directory of tokenizer')
    parser.add_argument('--model_max_length', default=2050, type=int, help='Model input size')
    parser.add_argument('--model_name', default="config/config_150M.json", type=str, help='Name of training model')
    parser.add_argument('--token_dropout', action='store_true', help='Token dropout')
    parser.add_argument('--positional_embedding_type', default="rotary", type=str, help='Positional embedding type rotary or absolute')
    parser.add_argument('--hidden_size', type=int, help='Hidden size of token')
    parser.add_argument('--intermediate_size', type=int, help='Intermediate size in Linear Module')
    parser.add_argument('--num_attention_heads', type=int, help='Number of attention heads')
    parser.add_argument('--num_hidden_layers', type=int, help='Num of hidden layers')
    parser.add_argument('--data_for_pretrain_path', type=str, help='Path of the data for pretrain')
    parser.add_argument('--flash_attention', action='store_true', help='Whether to use flash attention')
    parser.add_argument('--disable_tqdm', action='store_true', help='Whether to disable tqdm')
    parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 8)')
    parser.add_argument('--peak_lr', default=1e-4, type=float, help='Peak learning rate')
    parser.add_argument('--warmup_ratio', default=0.05, type=float, help='Warm up ratio')
    parser.add_argument('--max_steps', default=300000, type=int, help='Max training steps')
    parser.add_argument('--grad_clipping_norm', type=float, help='Max norm of the gradients in gradient clipping')
    parser.add_argument('--accum_steps', default=1, type=int, help='accumulation steps (default: 1)')
    parser.add_argument('--output_dir', type=str, help='Directory of training output')
    parser.add_argument('--save_steps', default=1000, type=int, help='Save steps')
    parser.add_argument('--logging_steps', default=100, type=int, help='when to compute the loss')
    parser.add_argument('--fp16', action='store_true', help='Training with fp16')
    parser.add_argument('--resume_training', action='store_true', help='Whether resume from training')
    parser.add_argument('--data_collator_patch', action='store_true', help='Whether use data collator patch')
    args = parser.parse_args()

    main(
        args.tokenizer_path, args.model_max_length, args.model_name, args.token_dropout, args.positional_embedding_type, args.hidden_size, 
        args.intermediate_size, args.num_attention_heads, args.num_hidden_layers, args.data_for_pretrain_path, args.flash_attention, args.disable_tqdm, 
        args.batch_size, args.peak_lr, args.warmup_ratio, args.max_steps, args.grad_clipping_norm, args.accum_steps, 
        args.output_dir, args.save_steps, args.logging_steps, args.fp16, args.resume_training, args.data_collator_patch
    )
