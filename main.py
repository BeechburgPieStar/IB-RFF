import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from utils.load_data import load_single_dataset
from utils.hsic import hsic_normalized_cca as hsic_fn
from backbones.FAN import FAN
from torchinfo import summary


def parse_args():
    parser = argparse.ArgumentParser(description="FARL with IB")
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--backbone', type=str, default="FAN")
    parser.add_argument('--bottleneck_type', type=str, default="DSCNN")
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--mlp_ratio', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dataset_name', type=str, default="ManyRx", choices=["ManyRx", "ManySig"])
    parser.add_argument('--exp', type=str, default="CRD", choices=["CRD", "CR"])
    parser.add_argument('--train_date', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--all_test_round', type=int, default=4)
    parser.add_argument('--test_round', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--pre_train', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--code_state', type=str, default="train_test", choices=["only_train", "only_test", "train_test"])
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def choose_backbone(backbone, bottleneck_type, tx_num, patch_size, embed_dim, mlp_ratio):
    backbone_map = {
        'FAN':    {'use_SAF': True,  'use_FAF': True},
        'FAN_A':  {'use_SAF': False, 'use_FAF': True},  # w/o SAF
        'FAN_B':  {'use_SAF': True,  'use_FAF': False},  # w/o FAF
        'FAN_C':  {'use_SAF': False,  'use_FAF': False}, # w/o SAF_FAF
    }

    if backbone not in backbone_map:
        raise ValueError(f"Unsupported backbone: {backbone}")

    config = backbone_map[backbone]

    return FAN(
        seq_len=256,
        in_chans=2,
        patch_size=patch_size,
        embed_dim=embed_dim,
        mlp_ratio=mlp_ratio,
        num_classes=tx_num,
        bottleneck_type=bottleneck_type,
        use_SAF=config['use_SAF'],
        use_FAF=config['use_FAF']
    )

def split_receivers(all_num=12, all_test_round=4, test_round=0):
    # 检查 test_round 是否超出范围
    if not (0 <= test_round < all_test_round):
        raise ValueError(f"Warning: test_round {test_round} is out of valid range [0, {all_test_round-1}].")
    
    if all_num % all_test_round != 0:
        raise ValueError(f"Warning: total rx number {all_num} is not divisible by test rounds {all_test_round}. The last round will have more elements.")

    receivers = list(range(all_num))
    per_round = all_num // all_test_round
    start = test_round * per_round
    end = all_num if test_round == all_test_round - 1 else start + per_round
    test = receivers[start:end]
    train = [r for r in receivers if r not in test]
    return train, test


def train(model, loss_fn, train_loader, optimizer, epoch, alpha):
    model.train()
    correct, total_loss = 0, 0
    total_hsic_data = 0
    total_samples = 0

    for data, target in train_loader:
        target = target.long()
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        hidden, output = model(data)
        output = F.log_softmax(output, dim=1)
        cls_loss = loss_fn(output, target)

        hidden_flattened = hidden.view(-1, np.prod(hidden.size()[1:]))
        data_flattened = data.view(-1, np.prod(data.size()[1:]))
        hsic_data = hsic_fn(hidden_flattened, data_flattened)

        total_hsic_data += hsic_data.item() * data.size(0)
        total_samples += data.size(0)
        loss = cls_loss + alpha*hsic_data
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    avg_hsic_data = total_hsic_data / total_samples
    acc = 100. * correct / len(train_loader.dataset)

    print(f"Train Epoch: {epoch} \tLoss: {total_loss / len(train_loader.dataset):.6f}, "
          f"Accuracy: {correct}/{len(train_loader.dataset)} ({acc:.2f}%), "
          f"Avg HSIC Data: {avg_hsic_data:.6f}")

def evaluate(model, loss_fn, val_loader, epoch):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in val_loader:
            target = target.long()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            _, output = model(data)
            output = F.log_softmax(output, dim=1)
            total_loss += loss_fn(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)
    print(f"\nValidation set: Loss: {avg_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({acc:.2f}%)\n")
    return avg_loss

def test(model, test_loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            target = target.long()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            _, output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    print(f"Test Accuracy: {correct / len(test_loader.dataset):.4f}")

def train_and_evaluate(model, loss_fn, train_loader, val_loader, optimizer, epochs, alpha, save_path):
    best_val_loss = float('inf')
    no_improve_count = 0
    patience_early_stop = 10
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)


    for epoch in range(1, epochs + 1):
        train(model, loss_fn, train_loader, optimizer, epoch, alpha)
        val_loss = evaluate(model, loss_fn, val_loader, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.6f} to {val_loss:.6f}, saving model.")
            best_val_loss = val_loss
            torch.save(model, save_path)
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"No improvement in validation loss for {no_improve_count} epoch(s).")

        if no_improve_count >= patience_early_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

        print("------------------------------------------------")

def prepare_dataset(dataset_name, rx_indexes, date_indexes, tx_num, is_eq, is_train, seed):
    x_all, y_all = [], []
    for rx_index in rx_indexes:
        for date_index in date_indexes:
            x, y = load_single_dataset(dataset_name, rx_index, date_index, tx_num, is_eq)
            x_all.append(x)
            y_all.append(y)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    if is_train:
        return train_test_split(x_all, y_all, test_size=0.3, random_state=seed)
    return x_all, y_all

def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu
    setup_seed(conf.seed)

    model_path = f'weights/{conf.backbone}_{conf.bottleneck_type}_{conf.patch_size}_{conf.embed_dim}_{conf.mlp_ratio}_{conf.alpha}_{conf.pre_train}_{conf.dataset_name}_{conf.train_date}_test_round={conf.test_round}_seed={conf.seed}.pth'
    tx_num, rx_num = (10, 32) if conf.dataset_name == "ManyRx" else (6, 12)

    rx_train, rx_test = split_receivers(rx_num, conf.all_test_round, conf.test_round)
    print(f"Train receivers: {rx_train}")
    print(f"Test receivers:  {rx_test}")

    x_train, x_val, y_train, y_val = prepare_dataset(conf.dataset_name, rx_train, conf.train_date, tx_num, 'non_equalized', True, conf.seed)
    train_loader = DataLoader(TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)), batch_size=conf.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val)), batch_size=conf.batch_size, shuffle=True)

    model = choose_backbone(conf.backbone, conf.bottleneck_type, tx_num, conf.patch_size, conf.embed_dim, conf.mlp_ratio).cuda()
    if conf.pre_train==1:
        pretrain_model_path = f'weights/{conf.backbone}_{conf.bottleneck_type}_{conf.patch_size}_{conf.embed_dim}_{conf.mlp_ratio}_0.0_0_{conf.dataset_name}_{conf.train_date}_test_round={conf.test_round}_seed={conf.seed}.pth'
        model = torch.load(pretrain_model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    loss_fn = nn.NLLLoss().cuda() if torch.cuda.is_available() else nn.NLLLoss()

    if conf.code_state in ["only_train", "train_test"]:
        train_and_evaluate(model, loss_fn, train_loader, val_loader, optimizer, conf.epochs, conf.alpha, model_path)

    if conf.code_state in ["only_test", "train_test"]:
        model = torch.load(model_path)
        if conf.exp == "CR":
            x_test, y_test = prepare_dataset(
                conf.dataset_name, rx_test, conf.train_date, tx_num, 
                'non_equalized', False, conf.seed
            )
            test_loader = DataLoader(
                TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), 
                batch_size=32, shuffle=True
            )
            print("------------------CR Testing------------------")
            test(model, test_loader)

        if conf.exp == "CRD":
            other_days = [date for date in [1, 2, 3, 4] if date not in conf.train_date]
            x_test, y_test = prepare_dataset(
                conf.dataset_name, rx_test, other_days, tx_num, 
                'non_equalized', False, conf.seed
            )
            test_loader = DataLoader(
                TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), 
                batch_size=32, shuffle=True
            )
            print("------------------CRD Testing------------------")
            test(model, test_loader)

if __name__ == '__main__':
    main()