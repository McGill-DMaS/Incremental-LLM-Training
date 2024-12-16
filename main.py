# Modified from https://github.com/karpathy/build-nanogpt

import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
import numpy as np
import wandb

from hellaswag import render_example, iterate_examples


n_stages = 4
# Set this to True to run the baseline with no freezing (no two-phase training)
no_freezing_baseline = False

wandb.init(project="nanogpt", name="incremental_llm"+"_"+str(n_stages)+"stages_no_freezing" if no_freezing_baseline else "incremental_llm"+"_"+str(n_stages)+"stages", resume='allow')
f_log = open("log_incremental_llm"+"_"+str(n_stages)+"stages"+"_log_no_freezing.txt" if no_freezing_baseline else "incremental_llm"+"_"+str(n_stages)+"stages"+"_log.txt", "w")

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # shift mask as well
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_stages: int = 20  # Number of stages


class IncrementalGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_stage = 1
        self.total_stages = config.n_stages
        self.layers_per_stage = config.n_layer // self.total_stages

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        num_layers_to_use = self.current_stage * self.layers_per_stage
        for i in range(num_layers_to_use):
            x = self.transformer.h[i](x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def set_stage(self, stage):
        assert 1 <= stage <= self.total_stages, "Stage must be within valid range"
        self.current_stage = stage

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = "/mnt/d/dataset/edu_fineweb100B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")
total_batch_size = 524288
B = 32
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

model = IncrementalGPT(GPTConfig(vocab_size=50304, n_stages=n_stages))
model.to(device)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

validation_interval = 250
hellaswag_interval = 250
current_overall_step = 0
max_steps = 10000

for stage in range(1, model.total_stages + 1):
    model.set_stage(stage)

    if not no_freezing_baseline:
        # Original behavior: freeze older layers in Phase 1
        for i in range((stage - 1) * model.layers_per_stage):
            for param in model.transformer.h[i].parameters():
                param.requires_grad = False

    if no_freezing_baseline:
        # Baseline: No Phase 1, directly train all layers
        if master_process:
            print(f"Stage {stage} (No freezing baseline): Training all layers immediately.")
        # Directly train all layers for the combined steps (equivalent to Phase 1+2)
        steps_per_stage = max_steps // model.total_stages
        for step in range(steps_per_stage):
            model.train()
            optimizer.zero_grad()
            loss_accum = 0.0
            t0 = time.time()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
            optimizer.step()
            current_overall_step += 1
            t1 = time.time()
            tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
            tokens_per_sec = tokens_processed / (t1 - t0)

            if master_process:
                print(f"Stage {stage} Step {step}: Loss = {loss_accum.item():.6f} | tok/sec: {tokens_per_sec:.2f}")
                wandb.log({"training_loss": loss_accum.item()}, step=current_overall_step)
                f_log.write(f"current_overall_step: {current_overall_step}  Stage {stage} Step {step}: Loss = {loss_accum.item():.6f} | tok/sec: {tokens_per_sec:.2f}\n")
                f_log.flush()

            # Validation
            if current_overall_step > validation_interval:
                model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(x, y)
                        val_loss_accum += loss.detach()
                    val_loss_accum /= val_loss_steps
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    print(f"Validation Loss at Stage {stage} Step {step}: {val_loss_accum.item():.4f}")
                    wandb.log({"validation_loss": val_loss_accum.item()}, step=current_overall_step)
                    f_log.write(f"current_overall_step: {current_overall_step} Validation Loss at Stage {stage} Step {step}: {val_loss_accum.item():.4f}\n")
                    f_log.flush()
                validation_interval += 250

            # HellaSwag
            if current_overall_step > hellaswag_interval:
                num_correct_norm = 0
                num_total = 0
                for i, example in enumerate(iterate_examples("val")):
                    if i % ddp_world_size != ddp_rank:
                        continue
                    _, tokens, mask, label = render_example(example)
                    tokens = tokens.to(device)
                    mask = mask.to(device)
                    with torch.no_grad():
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(tokens)
                        pred_norm = get_most_likely_row(tokens, mask, logits)
                    num_total += 1
                    num_correct_norm += int(pred_norm == label)
                if ddp:
                    num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                    num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                    dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                    dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                    num_total = num_total.item()
                    num_correct_norm = num_correct_norm.item()
                acc_norm = num_correct_norm / num_total
                if master_process:
                    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                    wandb.log({"hellaswag_accuracy": acc_norm}, step=current_overall_step)
                    f_log.write(f"current_overall_step: {current_overall_step} HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}\n")
                    f_log.flush()
                hellaswag_interval += 250
    else:
        # Original incremental approach with two phases
        # Phase 1: Train only the newly added layers
        if master_process:
            print(f"Starting Phase 1 of Stage {stage}")
        for step in range(max_steps // (2*model.total_stages)):
            model.train()
            optimizer.zero_grad()
            loss_accum = 0.0
            t0 = time.time()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
            optimizer.step()
            current_overall_step += 1
            t1 = time.time()
            tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
            tokens_per_sec = tokens_processed / (t1 - t0)

            # Validation, Logging, HellaSwag checks as before
            if current_overall_step > validation_interval:
                model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(x, y)
                        val_loss_accum += loss.detach()
                    val_loss_accum /= val_loss_steps
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    print(f"Validation Loss at Stage {stage} Phase 1 Step {step}: {val_loss_accum.item():.4f}")
                    wandb.log({"validation_loss": val_loss_accum.item()}, step=current_overall_step)
                    f_log.write(f"current_overall_step: {current_overall_step} Validation Loss at Stage {stage} Phase 1 Step {step}: {val_loss_accum.item():.4f}\n")
                    f_log.flush()
                validation_interval += 250

            if current_overall_step > hellaswag_interval:
                num_correct_norm = 0
                num_total = 0
                for i, example in enumerate(iterate_examples("val")):
                    if i % ddp_world_size != ddp_rank:
                        continue
                    _, tokens, mask, label = render_example(example)
                    tokens = tokens.to(device)
                    mask = mask.to(device)
                    with torch.no_grad():
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(tokens)
                        pred_norm = get_most_likely_row(tokens, mask, logits)
                    num_total += 1
                    num_correct_norm += int(pred_norm == label)
                if ddp:
                    num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                    num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                    dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                    dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                    num_total = num_total.item()
                    num_correct_norm = num_correct_norm.item()
                acc_norm = num_correct_norm / num_total
                if master_process:
                    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                    wandb.log({"hellaswag_accuracy": acc_norm}, step=current_overall_step)
                    f_log.write(f"current_overall_step: {current_overall_step} HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}\n")
                    f_log.flush()
                hellaswag_interval += 250

            if master_process:
                print(f"Stage {stage} Phase 1 Step {step}: Loss = {loss_accum.item():.6f} | tok/sec: {tokens_per_sec:.2f}")
                wandb.log({"training_loss": loss_accum.item()}, step=current_overall_step)
                f_log.write(f"current_overall_step: {current_overall_step}  Stage {stage} Phase 1 Step {step}: Loss = {loss_accum.item():.6f} | tok/sec: {tokens_per_sec:.2f}\n")
                f_log.flush()

        for i in range((stage - 1) * model.layers_per_stage):
            for param in model.transformer.h[i].parameters():
                param.requires_grad = True

        # Phase 2: Fine-tune all layers up to the current stage
        if master_process:
            print(f"Starting Phase 2 of Stage {stage}")
        for step in range(max_steps // (2*model.total_stages)):
            model.train()
            optimizer.zero_grad()
            loss_accum = 0.0
            t0 = time.time()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
            optimizer.step()
            current_overall_step += 1
            t1 = time.time()
            tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
            tokens_per_sec = tokens_processed / (t1 - t0)

            if master_process:
                print(f"Stage {stage} Phase 2 Step {step}: Loss = {loss_accum.item():.6f} | tok/sec: {tokens_per_sec:.2f}")
                wandb.log({"training_loss": loss_accum.item()}, step=current_overall_step)
                f_log.write(f"current_overall_step: {current_overall_step} Stage {stage} Phase 2 Step {step}: Loss = {loss_accum.item():.6f} | tok/sec: {tokens_per_sec:.2f}\n")
                f_log.flush()

        # Validation after Phase 2
        if current_overall_step > validation_interval:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    val_loss_accum += loss.detach()
                val_loss_accum /= val_loss_steps
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation Loss at Stage {stage} Phase 2: {val_loss_accum.item():.4f}")
                wandb.log({"validation_loss": val_loss_accum.item()}, step=current_overall_step)
                f_log.write(f"current_overall_step: {current_overall_step} Validation Loss at Stage {stage} Phase 2: {val_loss_accum.item():.4f}\n")
                f_log.flush()
            validation_interval += 250

        # HellaSwag after Phase 2
        if current_overall_step > hellaswag_interval:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                wandb.log({"hellaswag_accuracy": acc_norm}, step=current_overall_step)
                f_log.write(f"current_overall_step: {current_overall_step} HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}\n")
                f_log.flush()
            hellaswag_interval += 250

# Continual training after all stages
if master_process:
    print("Starting Continual Training with all parameters")

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

for step in range(max_steps):
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    optimizer.step()
    current_overall_step += 1
    t1 = time.time()
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)

    if master_process:
        print(f"Continual Training Step {step}: Loss = {loss_accum.item():.6f} | tok/sec: {tokens_per_sec:.2f}")
        wandb.log({"training_loss": loss_accum.item()}, step=current_overall_step)
        f_log.write(f"current_overall_step: {current_overall_step} Continual Training Step {step}: Loss = {loss_accum.item():.6f} | tok/sec: {tokens_per_sec:.2f}\n")
        f_log.flush()

    # Validation
    if current_overall_step > validation_interval:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                val_loss_accum += loss.detach()
            val_loss_accum /= val_loss_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Continual Training Validation Loss: {val_loss_accum.item():.4f}")
            wandb.log({"validation_loss": val_loss_accum.item()}, step=current_overall_step)
            f_log.write(f"current_overall_step: {current_overall_step} Continual Training Validation Loss: {val_loss_accum.item():.4f}\n")
            f_log.flush()
        validation_interval += 250

    # HellaSwag
    if current_overall_step > hellaswag_interval:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"Continual Training HellaSwag Accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
            wandb.log({"hellaswag_accuracy": acc_norm}, step=current_overall_step)
            f_log.write(f"current_overall_step: {current_overall_step} Continual Training HellaSwag Accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}\n")
            f_log.flush()
        hellaswag_interval += 250

f_log.close()
if ddp:
    destroy_process_group()
