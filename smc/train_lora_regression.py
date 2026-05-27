import math
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from transformers.trainer import unwrap_model
from peft import LoraConfig, get_peft_model, PeftModel
from scipy.stats import pearsonr, spearmanr
import mlflow
from transformers import EarlyStoppingCallback
import random

# utils
def world_size():
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def steps_per_epoch(n_examples, per_device_bs, grad_accum, world):
    return math.floor(n_examples / (per_device_bs * grad_accum * max(1, world)))


def type_bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# collator (float labels)
@dataclass
class FloatRegressionCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        labels = torch.tensor(
            [float(f["labels"]) for f in features], dtype=torch.float32
        )
        feats = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tokenizer.pad(
            feats,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


# model (LoRA base + regression head)
class LlamaWithLoraRegHead(nn.Module):
    def __init__(self, base, num_outputs: int = 1, loss_function: str = "mse", loss_weight: float = 0.0):
        super().__init__()
        self.base = base

        config = self.base.config
        if hasattr(config, "hidden_size"):
            hidden = config.hidden_size
        elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            # common for multimodal models
            hidden = config.text_config.hidden_size
        elif hasattr(config, "language_config") and hasattr(config.language_config, "hidden_size"):
            # another variation for multimodal models
            hidden = config.language_config.hidden_size
        elif hasattr(config, "d_model"):
            hidden = config.d_model
        elif hasattr(config, "n_embd"):
            hidden = config.n_embd
        else:            
            print("Available config attributes:", dir(config))
            raise AttributeError(f"Could not determine hidden size from config: {config}")

        base_dtype = next(self.base.parameters()).dtype
        self.head = nn.Linear(hidden, num_outputs, dtype=base_dtype)
        self.loss_choice = loss_function
        self.loss_weight = loss_weight # Store the multiplier

        ## debug: print loss choice
        print("Loss choice:", self.loss_choice)
        print("Loss weight:", self.loss_weight)
        print(f"Detected hidden size: {hidden}")

        
    @property
    def config(self):
        return self.base.config

    def gradient_checkpointing_enable(self, **kwargs):        
        try:
            self.base.config.use_cache = False
        except Exception:
            pass
        if hasattr(self.base, "gradient_checkpointing_enable"):
            return self.base.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base, "gradient_checkpointing_disable"):
            return self.base.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]  # [Batch, Tokens, Hidden size]
        mask = attention_mask.unsqueeze(
            -1
        )  # [B, T, 1]. This is 1 for tokens that are non-masked, 0 for masked tokens
        denom = mask.sum(dim=1).clamp_min(1)  # [B, 1]
        pooled = (last_hidden * mask).sum(
            dim=1
        ) / denom  # [B, H]. This results in the mean of the non-masked tokens.
        logits = self.head(pooled).squeeze(-1)  # [B]

        loss = None

        if labels is not None:
            labels = labels.float()
            weights = None
            if self.loss_weight != 0.0:
                # weight is an exponential function of the reward.
                weights = torch.exp(labels * self.loss_weight)
            if self.loss_choice == "bce":
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels, weight=weights, reduction='mean')
            else: # MSE
                sq_error = F.mse_loss(logits.float(), labels, reduction='none')
                if weights is not None:
                    sq_error = sq_error * weights # Apply weights only if they exist
                loss = sq_error.mean()
                
        return {"loss": loss, "logits": logits}

class PeftSavingTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        m = unwrap_model(self.model)

        # Save LoRA adapters (writes adapter_config.json + adapter_model.safetensors)
        if hasattr(m, "base") and isinstance(m.base, PeftModel):
            m.base.save_pretrained(output_dir)

        # Save regression head right next to the adapters
        if hasattr(m, "head"):
            torch.save(
                m.head.state_dict(), os.path.join(output_dir, "reg_head_state_dict.pt")
            )


# data loader
def load_regression_dataset(args, tokenizer):
    max_len = args.max_length
    ds = load_from_disk(args.input_dir)

    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_len)
        enc["labels"] = [float(x) for x in batch["reward"]]
        enc["t"] = batch["t"]
        return enc

    cols_to_keep = {"input_ids", "attention_mask", "labels", "t"}
    cols_in_ds = ds["train"].column_names if "train" in ds else ds["validation"].column_names
    cols_to_remove = [c for c in cols_in_ds if c not in cols_to_keep]

    ds = ds.map(preprocess, batched=True, remove_columns=cols_to_remove)

    return ds

@dataclass
class MetricsCollator:
    """callable class to compute regression and correlation metrics."""
    t: np.ndarray
    loss_function: str

    def __call__(self, eval_pred):
        preds, labels = eval_pred
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)

        metrics = {}
        metrics["mse"] = float(np.mean((preds - labels) ** 2))
        metrics["mae"] = float(np.mean(np.abs(preds - labels)))

        sigmoid_preds = 1 / (1 + np.exp(-preds))
        if self.loss_function == 'bce':
            epsilon = 1e-8            
            term1 = labels * np.log(sigmoid_preds + epsilon)
            term2 = (1 - labels) * np.log(1 - sigmoid_preds + epsilon)
            metrics["bce"] = -np.mean(term1 + term2)
        
            metrics["pearsonr_y"], _ = pearsonr(sigmoid_preds, labels)
            metrics["spearmanr_y"], _ = spearmanr(sigmoid_preds, labels)
            
        else:  # mse
            metrics["pearsonr_y"], _ = pearsonr(preds, labels)
            metrics["spearmanr_y"], _ = spearmanr(preds, labels)
                    
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-11B-Vision")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/webshop/Llama-3.2-11B-Vision-trajectories-idx_0-1500",
    )
    parser.add_argument("--output_dir", type=str, default="model/webshop/llama32-3b_reg_lora_ddp")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--num_outputs", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true"
    )  # default: False
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--loss_function",
        type=str,
        default="mse",
        choices=["mse", "bce"],
        help="Loss function to use ('mse' or 'bce').",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=1,
        help="Number of evaluation to wait for improvement before stopping. 0 to disable."
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="pearsonr_y",
        help="Metric to monitor for early stopping and saving the best model."
    )   
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.001,
        help="Minimum change in the monitored metric to be considered an improvement."
    )
    parser.add_argument(
        '--loss_weight',
        type=float,
        default=0.0,
        help="Controls exponential weighting. 0.0 disables it."
    )
    args = parser.parse_args()

    SEED = random.randrange(100)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    # mlflow — use file-based store (avoids SQLite schema conflicts)
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(args.output_dir)
    mlflow.start_run()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # dynamic pad to longest in batch

    # data
    ds = load_regression_dataset(args, tokenizer)
    eval_steps_t = np.array(ds["validation"]["t"])
    eval_metrics = MetricsCollator(t=eval_steps_t, loss_function=args.loss_function)

    # base model
    base_cfg = AutoConfig.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=base_cfg,
        torch_dtype=(torch.bfloat16 if type_bf16() else torch.float16),
        low_cpu_mem_usage=True,
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    lora_base = get_peft_model(base_model, lora_cfg)

    # freeze lm_head + embeddings
    for n, p in lora_base.named_parameters():
        if "lm_head" in n or "model.embed_tokens" in n:
            p.requires_grad = False

    # print trainable params
    lora_base.print_trainable_parameters()
    print("=" * 50)

    # wrap with regression head
    model = LlamaWithLoraRegHead(
        lora_base, 
        num_outputs=args.num_outputs, 
        loss_function=args.loss_function,
        loss_weight=args.loss_weight,
        )

    # steps per epoch (DDP aware)
    WORLD = world_size()
    eval_save_steps = steps_per_epoch(
        len(ds["train"]),
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        WORLD,
    )
    metric_name = args.metric_for_best_model
    greater_is_better = metric_name in ["pearsonr_y", "spearmanr_y"]

    # training args
    targs = TrainingArguments(
        output_dir=args.output_dir+"_SEED"+str(SEED),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_strategy="steps",
        logging_steps=int(
            1280
            // (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * max(1, WORLD)
            )
        ),
        report_to=["mlflow"],
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=args.save_total_limit,
        eval_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        bf16=type_bf16(),
        fp16=not type_bf16(),
        ddp_find_unused_parameters=True,
        dataloader_num_workers=WORLD,
        save_safetensors=False,
        greater_is_better=greater_is_better,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=False,
        seed=SEED,
        resume_from_checkpoint=True,
    )

    collator = FloatRegressionCollator(tokenizer)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold

    )

    trainer = PeftSavingTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=eval_metrics,
        callbacks=[early_stopping_callback]
    )

    if trainer.is_world_process_zero():
        print("Preparing 20% random subset for baseline evaluation...")
    
    val_full = ds["validation"]
    subset_size = int(len(val_full) * 0.2)
    val_subset = val_full.shuffle(seed=SEED).select(range(subset_size))
    
    subset_t = np.array(val_subset["t"])
    subset_metrics_collator = MetricsCollator(t=subset_t, loss_function=args.loss_function)

    original_computer = trainer.compute_metrics
    trainer.compute_metrics = subset_metrics_collator 

    if trainer.is_world_process_zero():
        print(f"Running baseline eval on {subset_size} examples...")
    
    baseline_metrics = trainer.evaluate(
        eval_dataset=val_subset,
        metric_key_prefix="eval_baseline" 
    )        
    trainer.compute_metrics = original_computer

    if trainer.is_world_process_zero():
        print("Baseline metrics:", baseline_metrics)
        mlflow.log_metrics(baseline_metrics, step=0)


    trainer.train()
    
    final_metrics = trainer.evaluate()
    trainer.save_metrics("eval", final_metrics)


if __name__ == "__main__":
    main()
