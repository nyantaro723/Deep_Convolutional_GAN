"""
DCGAN Inference Script

Generate images using a trained DCGAN model.

Usage:
    python generate.py --checkpoint checkpoints/generator_final.pth --num_images 16
"""

import argparse
import torch
from torchvision.utils import save_image
from dcgan import Generator


def generate_images(checkpoint_path, num_images=16, latent_dim=100, output_path='generated.png'):
    """
    Generate images using a trained generator.
    
    Args:
        checkpoint_path (str): Path to generator checkpoint
        num_images (int): Number of images to generate
        latent_dim (int): Dimension of latent vector
        output_path (str): Path to save generated images
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create and load generator
    generator = Generator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    
    print(f"Loaded generator from {checkpoint_path}")
    
    # Generate images
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        fake_images = generator(z)
    
    # Save images
    save_image(
        fake_images,
        output_path,
        normalize=True,
        nrow=int(num_images ** 0.5)
    )
    
    print(f"Generated {num_images} images and saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate images using trained DCGAN'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/generator_final.pth',
        help='Path to generator checkpoint (default: checkpoints/generator_final.pth)'
    )
    
    parser.add_argument(
        '--num_images',
        type=int,
        default=16,
        help='Number of images to generate (default: 16)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='generated.png',
        help='Output image path (default: generated.png)'
    )
    
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=100,
        help='Dimension of latent vector (default: 100)'
    )
    
    args = parser.parse_args()
    
    generate_images(
        args.checkpoint,
        num_images=args.num_images,
        latent_dim=args.latent_dim,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
