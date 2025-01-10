import torch
from torchvision import transforms
import cv2
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, mode="train"):
        """
        Initialize the dataset by loading file paths and defining transformations.
        Args:
            images_dir (str): Path to the dataset directory.
            mode (str): Either "train" or "test" to load respective data.
        """
        assert mode in ["train", "test"], "Mode must be either 'train' or 'test'"
        self.mode = mode

        # Define the transformation pipeline
        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load file paths based on mode
        if mode == "train":
            self.A_dir = sorted(glob(os.path.join(images_dir, "train_A", "*.jpg")))
            self.A_mask_dir = sorted(glob(os.path.join(images_dir, "train_A_mask", "*.png")))
            self.B_dir = sorted(glob(os.path.join(images_dir, "train_B", "*.jpg")))
            self.B_mask_dir = sorted(glob(os.path.join(images_dir, "train_B_mask", "*.png")))
        else:
            self.A_dir = sorted(glob(os.path.join(images_dir, "test_A", "*.jpg")))
            self.A_mask_dir = sorted(glob(os.path.join(images_dir, "test_A_mask", "*.png")))
            self.B_dir = sorted(glob(os.path.join(images_dir, "test_B", "*.jpg")))
            self.B_mask_dir = None

        # Ensure consistent dataset sizes
        if self.A_mask_dir:
            assert len(self.A_dir) == len(self.A_mask_dir), "Mismatch between A and A_mask lengths."
        if self.B_dir:
            assert len(self.A_dir) == len(self.B_dir), "Mismatch between A and B lengths."

    def create_empty_mask(self, shape):
        """Create an empty (white) mask when none is available"""
        return np.ones(shape, dtype=np.uint8) * 255

    def load_and_transform_image(self, image_path, is_mask=False):
        """Load and transform an image, with error handling"""
        if image_path is None and is_mask:
            # Create an empty mask if the mask path is None
            empty_mask = self.create_empty_mask((256, 256, 3))
            return self.to_tensor(empty_mask)
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.to_tensor(img)

    def __getitem__(self, idx):
        idx %= len(self.A_dir)  # Ensure index is valid
        
        # Load images and masks
        A_tensor = self.load_and_transform_image(self.A_dir[idx])
        
        A_mask_path = self.A_mask_dir[idx] if self.A_mask_dir else None
        A_mask_tensor = self.load_and_transform_image(A_mask_path, is_mask=True)
        
        B_tensor = self.load_and_transform_image(self.B_dir[idx])
        
        B_mask_path = self.B_mask_dir[idx] if self.B_mask_dir else None
        B_mask_tensor = self.load_and_transform_image(B_mask_path, is_mask=True) if self.mode == "train" else None

        # Return data based on mode
        if self.mode == "train":
            return {
                "A": A_tensor,
                "A_mask": A_mask_tensor,
                "B": B_tensor,
                "B_mask": B_mask_tensor
            }
        else:
            return {
                "A": A_tensor,
                "A_mask": A_mask_tensor,
                "B": B_tensor
            }

    def __len__(self):
        return len(self.A_dir)


def print_dataset(loader, mode="train"):
    print(f"---- {mode.upper()} DATASET ----")
    for batch_idx, batch in enumerate(loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  A shape: {batch['A'].shape}")
        print(f"  A_mask shape: {batch['A_mask'].shape}")
        print(f"  B shape: {batch['B'].shape}")
        if 'B_mask' in batch:
            print(f"  B_mask shape: {batch['B_mask'].shape}")
        else:
            print("  B_mask: None")

        if batch_idx == 1:
            break

def show_tensor(tensor):
    """
    Convert a tensor to a displayable image
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W)
    Returns:
        numpy array: Processed image ready for display
    """
    img = tensor.detach().cpu()
    img = img * 0.5 + 0.5  # Denormalize
    img = img.numpy().transpose(1, 2, 0)
    return np.clip(img, 0, 1)

def plot_batch_samples(batch):
    """
    Plot samples from a batch in an organized grid
    Args:
        batch (dict): Batch dictionary containing 'A', 'A_mask', 'B', and 'B_mask'
    """
    # Create figure with proper size and spacing
    fig = plt.figure(figsize=(16, 8))
    
    # Define the grid layout
    grid = plt.GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.2)
    
    # Plot original images in top row
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.imshow(show_tensor(batch['A'][0]))
    ax1.set_title('Input Image (A)', pad=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.imshow(show_tensor(batch['B'][0]))
    ax2.set_title('Target Image (B)', pad=10)
    ax2.axis('off')
    
    # Plot masks in bottom row
    ax3 = fig.add_subplot(grid[1, 0])
    ax3.imshow(show_tensor(batch['A_mask'][0]))
    ax3.set_title('Input Mask (A_mask)', pad=10)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(grid[1, 1])
    ax4.imshow(show_tensor(batch['B_mask'][0]))
    ax4.set_title('Target Mask (B_mask)', pad=10)
    ax4.axis('off')
    
    plt.suptitle('Dataset Sample Visualization', fontsize=16, y=1.02)
    plt.show()