"""
DCGAN (Deep Convolutional Generative Adversarial Network) Implementation

This module contains the Generator and Discriminator architectures for DCGAN.
Based on: "Unsupervised Representation Learning with Deep Convolutional 
Generative Adversarial Networks" (Radford et al., 2015)
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for DCGAN.
    
    Takes random noise (latent vector) as input and generates fake images.
    Architecture:
        - Input: Noise vector (batch_size, 100, 1, 1)
        - Output: Fake image (batch_size, 3, 64, 64)
    
    Args:
        latent_dim (int): Dimension of the latent noise vector. Default: 100
        feature_maps_gen (int): Number of feature maps in generator. Default: 64
    """
    
    def __init__(self, latent_dim=100, feature_maps_gen=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_maps_gen = feature_maps_gen
        
        # Layer 1: Fully connected layer
        # Input: (batch_size, latent_dim) -> Output: (batch_size, 512*4*4)
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Layer 2: Transposed Convolution (4x4 -> 8x8)
        self.tconv1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(256)
        
        # Layer 3: Transposed Convolution (8x8 -> 16x16)
        self.tconv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(128)
        
        # Layer 4: Transposed Convolution (16x16 -> 32x32)
        self.tconv3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(64)
        
        # Layer 5: Transposed Convolution (32x32 -> 64x64)
        self.tconv4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        """
        Forward pass of the generator.
        
        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            torch.Tensor: Generated image of shape (batch_size, 3, 64, 64)
        """
        # Fully connected layer
        x = self.fc(z)
        x = x.view(x.size(0), 512, 4, 4)  # Reshape to (batch_size, 512, 4, 4)
        
        # Transposed conv layers with batch norm and ReLU
        x = self.tconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.tconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.tconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Final layer with tanh activation (output range: [-1, 1])
        x = self.tconv4(x)
        x = self.tanh(x)
        
        return x


class Discriminator(nn.Module):
    """
    Discriminator network for DCGAN.
    
    Takes an image as input and outputs a probability that it's real.
    Architecture:
        - Input: Image (batch_size, 3, 64, 64)
        - Output: Scalar probability (batch_size, 1)
    
    Args:
        feature_maps_dis (int): Number of feature maps in discriminator. Default: 64
    """
    
    def __init__(self, feature_maps_dis=64):
        super(Discriminator, self).__init__()
        self.feature_maps_dis = feature_maps_dis
        
        # Layer 1: Convolution (64x64 -> 32x32)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        
        # Layer 2: Convolution (32x32 -> 16x16)
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(128)
        
        # Layer 3: Convolution (16x16 -> 8x8)
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(256)
        
        # Layer 4: Convolution (8x8 -> 4x4)
        self.conv4 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn4 = nn.BatchNorm2d(512)
        
        # Layer 5: Fully connected layer
        self.fc = nn.Linear(512 * 4 * 4, 1)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass of the discriminator.
        
        Args:
            x (torch.Tensor): Image of shape (batch_size, 3, 64, 64)
        
        Returns:
            torch.Tensor: Probability that image is real of shape (batch_size, 1)
        """
        # First convolution without batch norm
        x = self.conv1(x)
        x = self.leaky_relu(x)
        
        # Subsequent convolutions with batch norm and leaky ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


def weights_init(m):
    """
    Initialize weights of the network.
    
    This function initializes:
    - Convolutional and Transposed Convolutional layers: Normal distribution (mean=0, std=0.02)
    - Batch Normalization layers: Normal distribution (mean=1, std=0.02) for weight
                                  Constant (0) for bias
    
    Args:
        m (nn.Module): Module to initialize
    """
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        # Initialize Conv2d and ConvTranspose2d layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Initialize BatchNorm2d layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        # Initialize Linear layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    generator = Generator(latent_dim=100).to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, 100, device=device)
    
    print("Generator:")
    print(generator)
    print(f"\nGenerator input shape: {z.shape}")
    
    fake_images = generator(z)
    print(f"Generator output shape: {fake_images.shape}")
    
    print("\n" + "="*50 + "\n")
    
    print("Discriminator:")
    print(discriminator)
    print(f"\nDiscriminator input shape: {fake_images.shape}")
    
    output = discriminator(fake_images)
    print(f"Discriminator output shape: {output.shape}")
    print(f"Discriminator output (probabilities): {output.squeeze()}")
