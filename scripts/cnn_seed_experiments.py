import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torchvision.models import ResNet18_Weights

from torch.utils.data import DataLoader
from scripts.evaluation import calculate_metrics
from scripts.training import train_cnn

# cnn creation helper function
def create_cnn_model():
    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    original_conv = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )

    with torch.no_grad():
        model.conv1.weight[:] = original_conv.weight.mean(
            dim=1,
            keepdim=True
        )

    model.fc = nn.Linear(
        model.fc.in_features,
        1
    )

    return model

# reproducible loaders
def create_cnn_loaders(seed, train_dataset, val_dataset):
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=train_generator,
        num_workers=0
    )

    validation_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    return train_loader, validation_loader

# one full seed experiment
def run_cnn_seed(seed, train_dataset, val_dataset, weighted_loss, fixed_extractor_epochs=5, fine_tune_epochs=5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = create_cnn_model()

    seed_train_loader, seed_val_loader = create_cnn_loaders(seed, train_dataset, val_dataset)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    fixed_optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=0.001
    )

    fixed_train_losses, fixed_val_losses = train_cnn(
        model=model,
        train_loader=seed_train_loader,
        val_loader=seed_val_loader,
        criterion=weighted_loss,
        optimizer=fixed_optimizer,
        epochs=fixed_extractor_epochs
    )

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    fine_tune_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    fine_tune_optimizer = torch.optim.Adam(
        fine_tune_parameters,
        lr=0.0001
    )

    fine_tune_train_losses, fine_tune_val_losses = train_cnn(
        model=model,
        train_loader=seed_train_loader,
        val_loader=seed_val_loader,
        criterion=weighted_loss,
        optimizer=fine_tune_optimizer,
        epochs=fine_tune_epochs
    )

    model.eval()
    validation_probabilities = []
    validation_labels = []

    with torch.no_grad():
        for images, labels in seed_val_loader:
            logits = model(images).squeeze(1)
            probabilities = torch.sigmoid(logits)

            validation_probabilities.extend(
                probabilities.cpu().numpy()
            )
            validation_labels.extend(
                labels.cpu().numpy()
            )

    validation_probabilities = np.asarray(validation_probabilities)
    validation_labels = np.asarray(validation_labels)

    validation_predictions = (
        validation_probabilities >= 0.5
    ).astype(int)

    validation_metrics = calculate_metrics(
        labels=validation_labels,
        predictions=validation_predictions,
        probabilities=validation_probabilities
    )

    return {
        "seed": seed,
        "model": model,
        "validation_metrics": validation_metrics,
        "fixed_train_losses": fixed_train_losses,
        "fixed_val_losses": fixed_val_losses,
        "fine_tune_train_losses": fine_tune_train_losses,
        "fine_tune_val_losses": fine_tune_val_losses
    }