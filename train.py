import os, sys, string, random, time, uuid, json
import argparse
from typing import Dict, List, Optional, Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import wandb
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("high")

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from transformers import PretrainedConfig, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from datasets import load_dataset


class PretokenizedDataset(Dataset):
    def __init__(self, file_path, forecast):
        self.file_path = file_path
        self.forecast = forecast

        self.line_offsets = [0]
        self.length = 0
        with open(file_path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.length += 1
                self.line_offsets.append(f.tell())

            f.seek(self.line_offsets[0])
            context_length = len(f.readline().strip().split())
            if forecast:
                context_length //= 2

        rank_zero_info(f"Loaded a dataset {file_path}")
        rank_zero_info(f"  - Sequence length: {context_length}")
        rank_zero_info(f"  - Number of sequences: {self.length}")
        rank_zero_info(f"  - Number of tokens: {context_length*self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with open(self.file_path, "r", encoding="utf-8") as f:
            f.seek(self.line_offsets[idx])
            tokens = [int(token) for token in f.readline().strip().split()]

        if self.forecast:
            N = len(tokens) // 2
            input_ids = torch.tensor(tokens[:N], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = torch.tensor(tokens[N:], dtype=torch.long)
        else:
            input_ids = torch.tensor(tokens, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def read_file(filepath):
    all_tokens = (
        [f"c{i}" for i in range(6378)] + [f"d{i}" for i in range(8607)] + ["<sep>"]
    )
    tok2id = {all_tokens[idx]: idx for idx in range(len(all_tokens))}
    id2tok = {idx: all_tokens[idx] for idx in range(len(all_tokens))}

    with open(filepath, "r") as f:
        data = f.read()
    data = data.strip().split("\n")
    data = [line.split(",") for line in data]
    data = [[tok2id[tok] for tok in line] for line in data]
    return data


class ArxivPretokenizedDataset(Dataset):
    def __init__(self, citation_input_path, access_input_path, citation_target_path):
        self.citation_input_path = citation_input_path
        self.access_input_path = access_input_path
        self.citation_target_path = citation_target_path
        self.citation_input = read_file(self.citation_input_path)
        self.citation_target = read_file(self.citation_target_path)
        self.access_input = read_file(self.access_input_path)

        self.length = len(self.citation_input)
        assert len(self.citation_target) == self.length
        assert len(self.access_input) == self.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        labels = torch.tensor(self.citation_target[idx], dtype=torch.long)
        citation_input_ids = torch.tensor(self.citation_input[idx], dtype=torch.long)
        access_input_ids = torch.tensor(self.access_input[idx], dtype=torch.long)

        attention_mask = torch.ones_like(citation_input_ids)

        return {
            "citation_input_ids": citation_input_ids,
            "access_input_ids": access_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class GPT2LightningModule(pl.LightningModule):
    def __init__(
        self,
        data_dir: str = "",
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        pretrained_checkpoint: str = None,
        config: PretrainedConfig = None,
        forecast: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.forecast = forecast

        if pretrained_checkpoint:
            self.model = GPT2LMHeadModel.from_pretrained(pretrained_checkpoint)
            rank_zero_info(f"Loaded pre-trained model from {pretrained_checkpoint}")
        else:
            self.model = GPT2LMHeadModel(config)

        self.model.gradient_checkpointing_enable()

    def forward(
        self, citation_input_ids, access_input_ids, attention_mask=None, labels=None
    ):
        citation_embeds = self.model.transformer.wte(citation_input_ids)
        access_embeds = self.model.transformer.wte(access_input_ids)
        inputs_embeds = citation_embeds + access_embeds
        return self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            citation_input_ids=batch["citation_input_ids"],
            access_input_ids=batch["access_input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            citation_input_ids=batch["citation_input_ids"],
            access_input_ids=batch["access_input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        train_dataloader = self.train_dataloader()
        if hasattr(train_dataloader, "dataset"):
            dataset_size = len(train_dataloader.dataset)
            pct_start = min(0.3, self.hparams.warmup_steps / self.trainer.max_steps)

            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=self.trainer.max_steps,
                pct_start=pct_start,
                div_factor=100.0,
                final_div_factor=0.1,
                anneal_strategy="cos",
                three_phase=False,
                cycle_momentum=False,
            )

            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

            return [optimizer], [scheduler_config]

        return optimizer

    def train_dataloader(self):
        num_devices = max(1, self.trainer.num_devices)
        per_device_batch_size = self.hparams.train_batch_size // num_devices

        # dataset = PretokenizedDataset(
        #     os.path.join(self.data_dir, "train.txt"), self.forecast
        # )
        dataset = ArxivPretokenizedDataset(
            citation_input_path=os.path.join(self.data_dir, "citation_input_train.txt"),
            access_input_path=os.path.join(self.data_dir, "access_input_train.txt"),
            citation_target_path=os.path.join(
                self.data_dir, "citation_target_train.txt"
            ),
        )

        return DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        num_devices = max(1, self.trainer.num_devices)
        per_device_batch_size = self.hparams.train_batch_size // num_devices

        # dataset = PretokenizedDataset(
        #     os.path.join(self.data_dir, "valid.txt"), self.forecast
        # )
        dataset = ArxivPretokenizedDataset(
            citation_input_path=os.path.join(self.data_dir, "citation_input_val.txt"),
            access_input_path=os.path.join(self.data_dir, "access_input_val.txt"),
            citation_target_path=os.path.join(self.data_dir, "citation_target_val.txt"),
        )

        return DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )


class MaxStepProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self._persistent_bar = None

    def init_train_tqdm(self):
        if self._persistent_bar is None:
            bar = super().init_train_tqdm()
            bar.set_description("Training Progress")
            self._persistent_bar = bar

        return self._persistent_bar

    def on_train_epoch_start(self, trainer, pl_module):
        if self._persistent_bar is not None:
            self._persistent_bar.set_description("Training Progress")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current = trainer.global_step
        total = trainer.max_steps
        self.train_progress_bar.n = current
        self.train_progress_bar.total = total
        self._persistent_bar.refresh()


class HuggingFaceCheckpoint(ModelCheckpoint):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        if (trainer.global_step % self._every_n_train_steps) != 0:
            return

        step_dir = os.path.join(self.dirpath, f"step-{trainer.global_step}")
        os.makedirs(step_dir, exist_ok=True)

        model_path = os.path.join(step_dir, "model.pt")
        raw_model = pl_module.model if hasattr(pl_module, "model") else pl_module
        cpu_state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
        torch.save(cpu_state_dict, model_path)

        config_path = os.path.join(step_dir, "config.json")
        with open(config_path, "w") as f:
            if hasattr(self.config, "to_dict"):
                json.dump(self.config.to_dict(), f, indent=2)
            else:
                json.dump(self.config, f, indent=2)


def main(args):
    pl.seed_everything(args.seed)
    # wandb_logger = WandbLogger()

    # # Log hyperparameters to wandb
    # wandb_logger.log_hyperparams(vars(args))

    model_config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.seq_len,
        n_embd=args.hidden_dim,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        embd_pdrop=args.embed_pdrop,
        resid_pdrop=args.resid_pdrop,
        use_cache=False,
    )

    model = GPT2LightningModule(
        data_dir=args.data_dir,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        pretrained_checkpoint=args.checkpoint_path,
        config=model_config,
        forecast=args.forecast,
    )

    checkpoint_callback = HuggingFaceCheckpoint(
        config=model.model.config,
        dirpath=args.output_dir,
        filename="{step}",
        save_top_k=0,
        monitor=None,
        save_last=False,
        every_n_train_steps=args.steps_per_checkpoint,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"

    trainer = pl.Trainer(
        plugins=[],
        # plugins=[pl.plugins.environments.LightningEnvironment()],
        max_steps=args.num_train_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus_per_node,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=False),
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
            MaxStepProgressBar(),
        ],
        enable_progress_bar=True,
        precision="bf16-mixed" if args.bf16 else 32,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=10,
        # val_check_interval=args.steps_per_eval,
        val_check_interval=500,
        logger=pl.loggers.WandbLogger(
            name="logs", save_dir=args.output_dir, version=""
        ),
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT-2 training script using PyTorch Lightning"
    )
    parser.add_argument("--data_dir", type=str, help="Output directory for checkpoints")
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Initialize model weights from this checkpoint",
    )

    # Dataset parameters
    parser.add_argument("--forecast", action="store_true", help="Forecasting dataset")
    parser.add_argument(
        "--vocab_size", type=int, default=14986, help="Dataset vocabulary size"
    )
    parser.add_argument(
        "--seq_len", type=int, default=365, help="Dataset sequence length"
    )

    # Model parameters
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Model hidden dimensions"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Model number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Model number of layers"
    )
    parser.add_argument(
        "--embed_pdrop", type=float, default=0.1, help="Apply embedding dropout"
    )
    parser.add_argument(
        "--resid_pdrop", type=float, default=0.1, help="Apply residual dropout"
    )

    # Optimization parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_train_steps", type=int, default=50000, help="Number of training steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=200, help="Number of warmup steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 mixed precision training"
    )
    parser.add_argument(
        "--steps_per_eval",
        type=int,
        default=10000,
        help="Number of steps between validation evals",
    )
    parser.add_argument(
        "--steps_per_checkpoint",
        type=int,
        default=10000,
        help="Number of steps between checkpoints",
    )

    # System parameters
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--gpus_per_node", type=int, default=4, help="Number of GPUs per node"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
