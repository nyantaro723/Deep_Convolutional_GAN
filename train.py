"""
DCGAN Training Script

This script trains the DCGAN model on a specified dataset.
Supports CIFAR-10 and CelebA datasets.

Usage:
    python train.py --dataset cifar10 --batch_size 128 --num_epochs 100
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from dcgan import Generator, Discriminator, weights_init


def get_data_loader(dataset_name, batch_size, image_size=64):
    """
    Get data loader for the specified dataset.
    
    Args:
        dataset_name (str): Name of dataset ('cifar10' or 'celeba')
        batch_size (int): Batch size
        image_size (int): Image size (default: 64)
    
    Returns:
        DataLoader: Data loader for training
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                            std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    if dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == 'mnist':
        # For MNIST, convert to 3-channel
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                std=(0.5, 0.5, 0.5))
        ])
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader


def train_dcgan(args):
    """
    Train DCGAN model.
    
    Args:
        args: Arguments containing training configuration
    """
    # Create output directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    latent_dim = 100
    image_size = 64
    
    # Create models
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    lr = args.learning_rate
    beta1 = args.beta1
    
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Data loader
    dataloader = get_data_loader(args.dataset, args.batch_size, image_size)
    
    # Labels for real and fake
    real_label = 0.9  # Label smoothing
    fake_label = 0.1
    
    # Training loop
    g_losses = []
    d_losses = []
    
    print(f"Starting training for {args.num_epochs} epochs...")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    
    for epoch in range(args.num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)
            
            # Create labels
            real_labels = torch.full(
                (current_batch_size, 1), real_label, device=device
            )
            fake_labels = torch.full(
                (current_batch_size, 1), fake_label, device=device
            )
            
            # =====================
            # Train Discriminator
            # =====================
            optimizer_d.zero_grad()
            
            # Real images
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            z = torch.randn(current_batch_size, latent_dim, device=device)
            fake_images = generator(z)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            
            # =====================
            # Train Generator
            # =====================
            optimizer_g.zero_grad()
            
            # Generate new batch of fake images
            z = torch.randn(current_batch_size, latent_dim, device=device)
            fake_images = generator(z)
            
            # Generator wants discriminator to think fake images are real
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)
            
            g_loss.backward()
            optimizer_g.step()
            
            # Update epoch losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} "
                      f"G_loss: {g_loss.item():.4f}")
        
        # Average losses for the epoch
        epoch_d_loss /= len(dataloader)
        epoch_g_loss /= len(dataloader)
        
        g_losses.append(epoch_g_loss)
        d_losses.append(epoch_d_loss)
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] "
              f"Avg D_loss: {epoch_d_loss:.4f} "
              f"Avg G_loss: {epoch_g_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                generator.state_dict(),
                f'checkpoints/generator_epoch_{epoch+1}.pth'
            )
            torch.save(
                discriminator.state_dict(),
                f'checkpoints/discriminator_epoch_{epoch+1}.pth'
            )
            print(f"Saved checkpoint for epoch {epoch+1}")
        
        # Generate and save sample images
        if (epoch + 1) % args.save_interval == 0:
            with torch.no_grad():
                z = torch.randn(16, latent_dim, device=device)
                fake_images = generator(z)
                save_image(
                    fake_images,
                    f'outputs/generated_epoch_{epoch+1}.png',
                    normalize=True,
                    nrow=4
                )
    
    # Save final models
    torch.save(generator.state_dict(), 'checkpoints/generator_final.pth')
    torch.save(discriminator.state_dict(), 'checkpoints/discriminator_final.pth')
    
    # Plot and save loss curves
    plt.figure(figsize=(10, 5))
    epochs_range = range(1, len(g_losses) + 1)
    plt.plot(epochs_range, g_losses, label='Generator Loss', linewidth=2, marker='o')
    plt.plot(epochs_range, d_losses, label='Discriminator Loss', linewidth=2, marker='s')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('DCGAN Training Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/training_loss.png', dpi=150, bbox_inches='tight')
    print("Saved training loss plot to outputs/training_loss.png")
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Train DCGAN model'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'mnist'],
        help='Dataset to use (default: cifar10)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size (default: 128)'
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0002,
        help='Learning rate for Adam optimizer (default: 0.0002)'
    )
    
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.5,
        help='Beta1 parameter for Adam optimizer (default: 0.5)'
    )
    
    parser.add_argument(
        '--save_interval',
        type=int,
        default=10,
        help='Interval for saving checkpoints (default: 10)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Computing device (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Train the model
    train_dcgan(args)


if __name__ == '__main__':
    main()
