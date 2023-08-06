import argparse
import logging
import sys,os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.functional import f1_score
import random

from utils.data_loading import BasicDataset, CarvanaDataset, ImageDataset, collate_pool, HardAttenImageDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import numpy as np

dir_img = Path('/hardisk/image_process/generate_O1_O3_rocksalt/hardAttention_behindpart_angstrom_withNoise_4w_edge_dis_noCenter')
dir_label = Path('/hardisk/image_process/generate_O1_O3_rocksalt/hardAttention_behindpart_angstrom_withNoise_4w_edge_dis_noCenter')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints_hardAttention_behindpart_angstrom_withNoise_hasScheduler/')
if not os.path.exists(dir_checkpoint): os.makedirs(dir_checkpoint)


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              args=None):
    # 1. Create dataset

    dataset = ImageDataset(dir_img, dir_label)
    print('len)dataset:', len(dataset))
    # dataset = HardAttenImageDataset(dir_img, dir_label)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=12, pin_memory=True, collate_fn=collate_pool)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    print(len(train_set), len(val_set), len(train_loader), len(val_loader))

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    mean_acc_val_best = 0
    for epoch in range(0, epochs+1):
        net.train()
        F1_train = 0
        acc_train = 0
        epoch_loss = 0
        seed = random.randint(0, len(train_loader))
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for iter, (coors, labels, images) in enumerate(train_loader):
                # images = batch['image'].unsqueeze(dim=1)
                # true_masks = batch['label']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                # print(images.shape)
                labels = labels.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    # print('img shaoe:', images.shape)
                    # print('pre shape:', masks_pred.shape)
                    # break
                    labels_pred = []
                    for i in range(len(coors)):
                        # cor.append(coors[i][:,0])
                        # cor.append(coors[i][:,1])
                        # print(masks_pred[i].permute(1,2,0).shape)                        
                        h_idx = coors[i][:,0].clone().detach()
                        w_idx = coors[i][:,1].clone().detach()
                        # print(coors[i].shape)
                        # print(torch.max(h_idx), torch.min(h_idx), torch.max(w_idx), torch.min(w_idx))
                        # print(coors[i][:,0].shape, coors[i][:,1].shape)
                        labels_pred.append(masks_pred[i].clone().permute(1,2,0)[h_idx, w_idx])
                    # print('labels0: ', labels_pred[0].shape)
                    length_points = len(labels_pred[0])
                    labels_pred = torch.cat(labels_pred, dim=0)
                    # print(labels_pred.shape, labels.shape)
                    loss = criterion(labels_pred, labels)

                # calculate F1 score for this batch
                pre = nn.functional.softmax(labels_pred, dim=1))
                accuracy = torch.sum(torch.argmax(pre, dim=1)==labels)/len(pre)
                acc_train = acc_train + accuracy
                F1Score = f1_score(pre, labels, num_classes=args.classes, average='macro')
                F1_train = F1_train + F1Score.item()
                # save pre and label
                if epoch%10 == 0 and iter == seed:
                    # print(length_points)
                    coors_label = torch.cat((coors[0].to(device=device), torch.argmax(pre, dim=1)[:length_points].unsqueeze(dim=1),\
                         labels[:length_points].unsqueeze(dim=1)), dim=1)
                    np.savetxt(os.path.join(dir_checkpoint, 'epoch_{}_pre_true_train.txt'.format(epoch)) ,np.array(coors_label.cpu()))
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                #remove chche
                del coors, labels, images
                # torch.cuda.empty_cache()
            print('mean_F1_train: {:.9f}'.format(F1_train/len(train_loader)))
            print('mean_acc_train: {:.9f}'.format(acc_train/len(train_loader)))

        #evaluate on validation set
        net.eval()
        F1_val = 0
        acc_val = 0
        seed = random.randint(0, int(len(val_loader)/batch_size))
        with tqdm(total=n_val, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for iter, (coors, labels, images) in enumerate(val_loader):
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)

                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(images)
                        labels_pred = []
                        for i in range(len(coors)):                       
                            h_idx = coors[i][:,0].clone().detach()
                            w_idx = coors[i][:,1].clone().detach()
                            labels_pred.append(masks_pred[i].clone().permute(1,2,0)[h_idx, w_idx])
                        length_points = len(labels_pred[0])
                        labels_pred = torch.cat(labels_pred, dim=0)
                        loss = criterion(labels_pred, labels)
                    pre = nn.functional.softmax(labels_pred, dim=1)
                    accuracy = torch.sum(torch.argmax(pre, dim=1)==labels)/len(pre)
                    acc_val = acc_val + accuracy
                    # print(pre.shape, labels.shape)
                    F1Score = f1_score(pre, labels, num_classes=args.classes, average='macro')
                    F1_val = F1_val + F1Score.item()
                if epoch%10 == 0 and iter == seed:
                    # print(length_points)
                    coors_label = torch.cat((coors[0].to(device=device), torch.argmax(pre, dim=1)[:length_points].unsqueeze(dim=1),\
                         labels[:length_points].unsqueeze(dim=1)), dim=1)
                    np.savetxt(os.path.join(dir_checkpoint, 'epoch_{}_pre_true_val.txt'.format(epoch)) ,np.array(coors_label.cpu()))
                #remove cache
                del coors, labels, images
                # torch.cuda.empty_cache()
            print('mean_F1_val: {}'.format(F1_val/len(val_loader)))
            mean_acc_val = acc_val/len(val_loader)
            print('mean_acc_val: {}'.format(mean_acc_val))
        scheduler.step(mean_acc_val)
        if mean_acc_val > mean_acc_val_best:
            mean_acc_val_best = mean_acc_val
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_best.pth'.format(epoch)))
            logging.info(f'Best Checkpoint {epoch} saved!')
        if save_checkpoint:
            if epoch %100 == 0:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
        torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  args = args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
