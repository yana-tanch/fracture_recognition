import torch
import torch.nn as nn
import hydra
import numpy as np
import pyrootutils

from tqdm import tqdm
from dvclive import Live
from torch.utils.data import DataLoader

pyrootutils.setup_root(__file__, indicator='.project', pythonpath=True)

from src.utils import make_deterministic, get_device
from src.model import make_anchors, ModelYOLO
from src.dataset import get_datasets
from src.loss import Loss


def build_optimizer(params, model):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()

    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    optimizer = torch.optim.SGD(
        g[2],
        lr=params.optimizer.lr,
        momentum=params.optimizer.momentum,
        nesterov=params.optimizer.nesterov)

    accumulate = max(round(params.nbs / params.batch_size), 1)
    weight_decay = params.optimizer.weight_decay * params.batch_size * accumulate / params.nbs  # scale weight_decay

    optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)

    return optimizer


@hydra.main(version_base='1.3', config_path='../conf', config_name='config.yaml')
def main(cfg):
    params = cfg.train
    number_of_classes = len(params.labels)

    make_deterministic(params.seed)
    device = get_device(params.device)

    model = ModelYOLO(nc=number_of_classes).to(device)

    optimizer = build_optimizer(params, model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=params.scheduler.T_0,
        T_mult=params.scheduler.T_mult
    )

    train_dataset, valid_dataset = get_datasets(params)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False
    )

    anchors, strides = make_anchors(device)
    loss_fn = Loss(params, number_of_classes, anchors, strides, device)

    accumulate = max(round(params.nbs / params.batch_size), 1)
    last_opt_step = -1

    with Live(dir='dvclive/train', save_dvc_exp=False) as live:
        for epoch in tqdm(range(1, params.epochs + 1), leave=False, desc='epoch'):
            # TRAINING LOOP
            model.train()
            optimizer.zero_grad()

            train_losses = np.zeros((0,))
            number_of_batches = len(train_loader)

            progress = tqdm(enumerate(train_loader), leave=False)

            for i, (batch, targets) in progress:
                ni = i + number_of_batches * epoch

                batch = batch.float().unsqueeze(1).to(device) / 255
                targets = targets.float().to(device)

                predictions = model(batch)
                loss = loss_fn(predictions, targets)

                loss.backward()

                if ni - last_opt_step >= accumulate:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()
                    last_opt_step = ni

                train_losses = np.append(train_losses, loss.item())

                progress.set_description(
                    f'train epoch: {epoch} / {params.epochs} '
                    f'loss: {train_losses.mean():.4f}'
                )

            scheduler.step()

            # VALIDATION LOOP
            model = model.eval()
            valid_losses = np.zeros((0,))

            progress = tqdm(valid_loader, leave=False)

            for batch, targets in progress:
                batch = batch.float().unsqueeze(1).to(device) / 255
                targets = targets.float().to(device)

                with torch.no_grad():
                    predictions = model(batch)
                    loss = loss_fn(predictions, targets)

                valid_losses = np.append(valid_losses, loss.item())

                progress.set_description(
                    f'valid epoch: {epoch} / {params.epochs} '
                    f'loss: {valid_losses.mean():.4f}'
                )

            # SAVE MODEL
            torch.save({'model_state_dict': model.state_dict()}, params.model)

            train_loss = train_losses.mean()
            valid_loss = valid_losses.mean()

            live.log_metric('train_loss', train_loss)
            live.log_metric('valid_loss', valid_loss)
            live.log_metric('lr', scheduler.get_last_lr()[0])

            live.next_step()


if __name__ == '__main__':
    main()
