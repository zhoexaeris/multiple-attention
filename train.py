import os
import time
import logging
import warnings
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from models.MAT import MAT
from datasets.dataset import DeepfakeDataset
from AGDA import AGDA
import cv2
from utils import dist_average, ACC

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Ensure GPUs are available
assert torch.cuda.is_available()

print("[DEBUG] train.py started...")

def load_state(net, ckpt):
    """ Load model weights from checkpoint """
    sd = net.state_dict()
    nd = {}
    goodmatch = True

    for i in ckpt:
        if i in sd and sd[i].shape == ckpt[i].shape:
            nd[i] = ckpt[i]
        else:
            print(f"Failed to load: {i}")
            goodmatch = False

    net.load_state_dict(nd, strict=False)
    return goodmatch

def train_loop(net, train_loader, optimizer, scheduler, config, device):
    """Runs the training loop."""
    net.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = F.cross_entropy(outputs, targets)  # Compute loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)

        if batch_idx % 10 == 0:
            print(f"[DEBUG] Batch {batch_idx}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    acc = correct / total_samples

    print(f"[INFO] Training Loss: {avg_loss:.4f}, Accuracy: {acc:.4%}")
    scheduler.step()


def main_worker(local_rank, world_size, rank_offset, config):
    print(f"[DEBUG] Using dataset label: {config.datalabel}")

    rank = local_rank + rank_offset

    print(f"[DEBUG] main_worker started! Rank: {rank}, Local Rank: {local_rank}")

    # ✅ Only initialize distributed training if using multiple GPUs
    if world_size > 1:
        print(f"[DEBUG] Initializing distributed training with {world_size} GPUs...")
        dist.init_process_group(backend='gloo', init_method=config.url, world_size=world_size, rank=rank)
    else:
        print("[DEBUG] Running on a single GPU: No distributed training initialized.")

    # ✅ Set device manually
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # ✅ Load Dataset
    train_dataset = DeepfakeDataset(phase='train', datalabel=config.datalabel, resize=config.resize, normalize=config.normalize)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.workers)

    print(f"[DEBUG] Train dataset loaded: {len(train_dataset)} samples")

    # ✅ Load Model
    net = MAT(**config.net_config).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, betas=config.adam_betas, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)

    # ✅ Start Training
    print("[DEBUG] Training starts...")
    train_loop(net, train_loader, optimizer, scheduler, config, local_rank)


def train_loss(loss_pack, config):
    """ Calculate total loss """
    if 'loss' in loss_pack:
        return loss_pack['loss']

    loss = config.ensemble_loss_weight * loss_pack['ensemble_loss'] + config.aux_loss_weight * loss_pack['aux_loss']
    
    if config.AGDA_loss_weight != 0:
        loss += config.AGDA_loss_weight * loss_pack['AGDA_ensemble_loss'] + config.match_loss_weight * loss_pack['match_loss']

    return loss

def run(logs, data_loader, net, optimizer, local_rank, config, AG=None, phase='train'):
    """ Train or Validate the Model """
    if local_rank == 0:
        print(f"Starting {phase}...")

    if config.AGDA_loss_weight == 0:
        AG = None

    recorder = {}
    record_list = ['ensemble_loss', 'aux_loss', 'ensemble_acc']
    if AG is not None:
        record_list += ['AGDA_ensemble_loss', 'match_loss']

    for i in record_list:
        recorder[i] = dist_average(local_rank)

    start_time = time.time()
    
    if phase == 'train':
        net.train()
    else:
        net.eval()

    for i, (X, y) in enumerate(data_loader):
        X = X.to(local_rank, non_blocking=True)
        y = y.to(local_rank, non_blocking=True)

        with torch.set_grad_enabled(phase == 'train'):
            loss_pack = net(X, y, train_batch=True, AG=AG)

        if phase == 'train':
            batch_loss = train_loss(loss_pack, config)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            loss_pack['ensemble_acc'] = ACC(loss_pack['ensemble_logit'], y)

        for i in record_list:
            recorder[i].step(loss_pack[i])

    batch_info = [f"{i}:{recorder[i].get():.4f}" for i in record_list]
    end_time = time.time()

    if local_rank == 0:
        logging.info(f"{phase}: {'  '.join(batch_info)}, Time {end_time - start_time:.2f}")


def distributed_train(config):
    print("[DEBUG] Running training on a single GPU.")
    
    # Instead of using multiprocessing, directly call main_worker()
    main_worker(0, 1, 0, config)  # (local_rank=0, world_size=1, rank_offset=0, config=config)

    print("[DEBUG] Training completed successfully.")

