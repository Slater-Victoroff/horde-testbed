from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


class RGBImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_to_grayscale = RGBToGrayscaleImageNet()  # Shared grayscale representation
        self.grayscale_to_rgb = GrayscaleToRGBImageNet()  # Control determines RGB reconstruction

    def forward(self, rgb, control):
        # Compute a single shared grayscale representation
        B, C, H, W = rgb.shape
        grayscale = self.rgb_to_grayscale(rgb)  # Grayscale for each image: [B, 1, H, W]

        # Mean pool across the batch to create a shared grayscale representation
        shared_grayscale = grayscale.mean(dim=0, keepdim=True)  # Shared grayscale: [1, 1, H, W]

        shared_grayscale = (shared_grayscale - shared_grayscale.mean()) / (shared_grayscale.std() + 1e-5)

        # Expand the shared grayscale back to the batch dimension
        shared_grayscale = shared_grayscale.expand(B, -1, -1, -1)  # [B, 1, H, W]

        # Reconstruct RGB using the control parameter
        reconstructed_rgb = self.grayscale_to_rgb(shared_grayscale, control)  # [B, 3, H, W]

        return shared_grayscale, reconstructed_rgb


class RGBToGrayscaleImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Input: [B, 3, H, W]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # Output: [B, 1, H, W]
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass through the encoder
        return self.encoder(x)  # Output: [B, 1, H, W]


class GrayscaleToRGBImageNet(nn.Module):
    def __init__(self, control_dim=2):
        super().__init__()
        self.control_dim = control_dim

        # Process grayscale image
        self.grayscale_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Process control parameter
        self.control_branch = nn.Sequential(
            nn.Linear(control_dim, 32),
            nn.ReLU(),
        )

        # Combine grayscale and control features
        self.decoder = nn.Sequential(
            nn.Conv2d(32 + 32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, grayscale, control):  # grayscale: [B, 1, H, W], control: [B, C]
        B, _, H, W = grayscale.shape
        # Process grayscale image
        grayscale_features = self.grayscale_branch(grayscale)  # [B, 32, H, W]

        # Process control parameter
        control_features = self.control_branch(control)  # [B, 32]
        control_features = control_features.view(B, 32, 1, 1).expand(-1, -1, H, W)  # [B, 32, H, W]

        modulated_features = grayscale_features * control_features

        # Combine features
        combined_features = torch.cat([modulated_features, control_features], dim=1)  # [B, 64, H, W]

        # Decode to RGB
        reconstructed_rgb = self.decoder(combined_features)  # [B, 3, H, W]

        return reconstructed_rgb


def train_image_model(model, test_images, criterion, optimizer, device='cuda', epochs=1000, shuffle=True, batch_size=4096):
    rgb_tensor = load_patch_images(test_images, device)
    dataset = torch.utils.data.TensorDataset(rgb_tensor, rgb_tensor)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        # Training loop
        for rgb, target_rgb in train_dataloader:
            batch_size = rgb.size(0)

            # Create one-hot encoding for the batch, with ones on the diagonal
            control = torch.eye(batch_size, device=rgb.device)  # Identity matrix as one-hot encoding

            # Forward pass
            shared_grayscale, reconstructed_rgb = model(rgb, control)
            loss = criterion(reconstructed_rgb, target_rgb)  # Scale loss for better convergence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Last learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                save_debug_images(shared_grayscale, reconstructed_rgb, target_rgb, epoch + 1)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.6f}")


def save_debug_images(shared_grayscale, reconstructed_rgb, target_rgb, epoch, output_dir="debug_outputs"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save shared grayscale (same for all entries)
    grayscale_image = shared_grayscale[0].squeeze(0)  # Take the first image and remove channel dim: [H, W]
    grayscale_path = os.path.join(output_dir, f"shared_grayscale_epoch_{epoch}.png")
    save_image(grayscale_image, grayscale_path)
    print(f"Saved shared grayscale to {grayscale_path}")

    # Save reconstructed RGB and target RGB for each entry
    for i in range(reconstructed_rgb.size(0)):
        # Save reconstructed RGB
        reconstructed_image = reconstructed_rgb[i]  # Take the i-th image: [3, H, W]
        reconstructed_path = os.path.join(output_dir, f"reconstructed_rgb_{i}_epoch_{epoch}.png")
        save_image(reconstructed_image, reconstructed_path)
        print(f"Saved reconstructed RGB to {reconstructed_path}")


def load_patch_images(image_paths, device):
    rgb_tensors = []

    for image_path in image_paths:
        rgb_image = Image.open(image_path).convert("RGB")
        rgb_tensor = transforms.ToTensor()(rgb_image).unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
        rgb_tensors.append(rgb_tensor)

    combined_rgb_tensor = torch.cat(rgb_tensors, dim=0).to(device)  # Combine into a batch: [B, 3, H, W]
    return combined_rgb_tensor
