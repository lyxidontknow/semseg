import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model.pspnet import PSPNet
from pytorch.benchmarks.dynamo.check_accuracy import check_accuracy
from utils import get_loaders

import numpy as np
import random


def train_fn(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)
    total_loss = 0.0
    for index, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.unsqueeze(1).float().to(DEVICE)

        with torch.cuda.amp.autocast():
            predict = model(data)
            loss = loss_fn(predict, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())
    return total_loss/len(loader)


def check_accuracy(loader, model, DEVICE="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.unsqueeze(1).to(DEVICE)
            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()
            num_correct += (predictions == y).sum()
            num_pixels += torch.numel(predictions)
            dice_score += (2*(predictions*y).sum())/(2*(predictions*y).sum()+((predictions*y) < 1).sum())

    accuracy = round(float(num_correct/num_pixels), 4)
    dice = round(float(dice_score)/len(loader), 4)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100:.2f}")
    print(f"Dice score:{dice_score/len(loader)}")

    model.train()

    return accuracy, dice


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],)

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], )

    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
                                           train_transform, val_transform, BATCH_SIZE, NUM_WORKERS,
                                           PIN_MEMORY)
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=1, use_ppm=True, pretrained=False)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for index in range(NUM_EPOCH):
        print("current epoch: ", index)
        train_loss = train_fn(train_loader, model, loss_fn, optimizer, scaler)
        train_losses.append(train_loss)

        accuracy, dice = check_accuracy(train_loader, model, DEVICE="cuda")
        val_acc.append(accuracy)
        val_dice.append(dice)


if __name__ == '__main__':
    # Hyperparameters

    LEARNING_RATE = 1e-8
    BATCH_SIZE = 4
    NUM_EPOCH = 2
    NUM_WORKERS = 2
    PIN_MEMORY = True
    LOAD_MODEL = False
    DEVICE = "cuda"
    print(DEVICE)

    TRAIN_IMG_DIR = "./data/gtFine_trainvaltest/getFine/train/aachen"
    TRAIN_MASK_DIR = "./data/gtFine_trainvaltest/getFine/train/aachen"
    VAL_IMG_DIR = "./data/gtFine_trainvaltest/getFine/val/frankfurt"
    VAL_MASK_DIR = "./data/gtFine_trainvaltest/getFine/val/frankfurt"

    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 240

    train_losses = []
    val_acc = []
    val_dice = []
    main()
