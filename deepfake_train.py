import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import numpy.linalg as np_linalg
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
# Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = []
        self._find_images(image_folder)

    def _find_images(self, folder):
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isdir(filepath):
                self._find_images(filepath)
            elif filename.endswith(".jpg") or filename.endswith(".png"):
                self.image_files.append(filepath)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Resize to 224x224
        image = np.array(image)

        # Perform SVD
        svd_channels = []
        for channel in range(3):
            u, s, v = np_linalg.svd(image[:, :, channel], full_matrices=False)
            svd_channels.append((u, s, v))

        # Retain 80% energy
        num_singular_values = np.argmax(
            np.cumsum(svd_channels[0][1]) / np.sum(svd_channels[0][1]) >= 0.8
        )

        # Reconstruct 80% energy image
        low_rank_img = np.zeros_like(image)
        for channel in range(3):
            u, s, v = svd_channels[channel]
            low_rank_img[:, :, channel] = np.dot(
                u[:, : num_singular_values + 1],
                np.dot(np.diag(s[: num_singular_values + 1]), v[: num_singular_values + 1, :]),
            )

        # Normalize images
        low_rank_img = np.clip(low_rank_img, 0, 255).astype(np.uint8) / 255.0
        target_img = image / 255.0

        # Convert to PyTorch tensor format (C, H, W)
        low_rank_img = torch.tensor(low_rank_img.transpose((2, 0, 1)), dtype=torch.float)
        target_img = torch.tensor(target_img.transpose((2, 0, 1)), dtype=torch.float)

        return low_rank_img, target_img
    
class UNetVAE(nn.Module):
    def __init__(self, input_size=224):
        super(UNetVAE, self).__init__()

        # Encoder (Downsampling Path)
        self.enc1 = self.conv_block(3, 64)   # 224x224 -> 112x112
        self.enc2 = self.conv_block(64, 128) # 112x112 -> 56x56
        self.enc3 = self.conv_block(128, 256) # 56x56 -> 28x28
        self.enc4 = self.conv_block(256, 512) # 28x28 -> 14x14
        self.enc5 = self.conv_block(512, 1024) # 14x14 -> 7x7
        self.enc6 = self.conv_block(1024, 2048) # 7x7 -> 3x3

        # Determine dynamically correct `flatten_dim`
        dummy_input = torch.randn(1, 3, input_size, input_size)  # Batch size 1
        with torch.no_grad():
            dummy_output = self.enc6(self.enc5(self.enc4(self.enc3(self.enc2(self.enc1(dummy_input))))))
            self.flatten_dim = dummy_output.numel()  # Compute the exact size

        self.fc_mu = nn.Linear(self.flatten_dim, 128)
        self.fc_logvar = nn.Linear(self.flatten_dim, 128)
        self.decoder_input = nn.Linear(128, self.flatten_dim)

        # Decoder (Upsampling Path with Skip Connections)
        self.dec6 = self.upconv_block(2048 + 2048, 1024)
        self.dec5 = self.upconv_block(1024 + 1024, 512)
        self.dec4 = self.upconv_block(512 + 512, 256)
        self.dec3 = self.upconv_block(256 + 256, 128)
        self.dec2 = self.upconv_block(128 + 128, 64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels):
        """Encoder Convolution Block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Downsampling
        )

    def upconv_block(self, in_channels, out_channels):
        """Decoder Upsampling Block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )

    def encode(self, x):
        """Encoder Forward Pass with Skip Connections"""
        x1 = self.enc1(x)  # 224x224 -> 112x112
        x2 = self.enc2(x1) # 112x112 -> 56x56
        x3 = self.enc3(x2) # 56x56 -> 28x28
        x4 = self.enc4(x3) # 28x28 -> 14x14
        x5 = self.enc5(x4) # 14x14 -> 7x7
        x6 = self.enc6(x5) # 7x7 -> 3x3

        x_flattened = x6.view(x6.size(0), -1)  # Flatten dynamically

        mean = self.fc_mu(x_flattened)
        log_var = self.fc_logvar(x_flattened)
        return mean, log_var, (x6, x5, x4, x3, x2, x1)

    def reparameterize(self, mean, log_var):
        """Reparameterization Trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, skip_connections):
        """Decoder Forward Pass with Skip Connections"""
        x6, x5, x4, x3, x2, x1 = skip_connections  # Retrieve skip connections
        
        # Dynamically infer spatial dimensions
        batch_size, channels, h, w = x6.shape
        z = self.decoder_input(z)  # Fully connected layer output (batch_size, flatten_dim)
        z = z.view(batch_size, channels, h, w)  # Ensure correct shape

        z = torch.cat((z, x6), dim=1)
        z = self.dec6(z)
        z = nn.functional.interpolate(z, size=(7, 7), mode='nearest') # upsample z to match the spatial dimensions of x5
        z = torch.cat((z, x5), dim=1)
        z = self.dec5(z)
        z = nn.functional.interpolate(z, size=(14, 14), mode='nearest') # upsample z to match the spatial dimensions of x4
        z = torch.cat((z, x4), dim=1)
        z = self.dec4(z)
        z = nn.functional.interpolate(z, size=(28, 28), mode='nearest') # upsample z to match the spatial dimensions of x3
        z = torch.cat((z, x3), dim=1)
        z = self.dec3(z)
        z = nn.functional.interpolate(z, size=(56, 56), mode='nearest') # upsample z to match the spatial dimensions of x2
        z = torch.cat((z, x2), dim=1)
        z = self.dec2(z)
        z = nn.functional.interpolate(z, size=(112, 112), mode='nearest') # upsample z to match the spatial dimensions of x1
        z = torch.cat((z, x1), dim=1)
        z = self.dec1(z)

        return z

    def forward(self, x):
        mean, log_var, skip_connections = self.encode(x)
        z = self.reparameterize(mean, log_var)
        output = self.decode(z, skip_connections)
        return output, mean, log_var

# Training configuration
# real_train_path = 'DeepFake/DFDC/train/real'
# fake_train_path = 'DeepFake/DFDC/train/fake'
# real_test_path = 'DeepFake/DFDC/test/real'
# fake_test_path = 'DeepFake/DFDC/test/fake'
real_train_path = 'DeepFake/FaceForensics++/original_sequences_split/youtube/train/c23/frames'
fake_train_path  = 'DeepFake/FaceForensics++/manipulated_sequences_split/FaceSwap/train/c23/frames'
real_test_path = 'DeepFake/FaceForensics++/original_sequences_split/youtube/test/c23/frames'
fake_test_path = 'DeepFake/FaceForensics++/manipulated_sequences_split/FaceSwap/test/c23/frames'
batch_size = 128
num_epochs = 15
learning_rate = 0.001
l1_lambda = 0.65 # L1 regularization strength

# Create dataset and data loader
real_train_dataset = ImageDataset(real_train_path)
train_loader = DataLoader(real_train_dataset, batch_size=batch_size, shuffle=True)

real_test_dataset = ImageDataset(real_test_path)
fake_test_dataset = ImageDataset(fake_test_path)
real_test_loader = DataLoader(real_test_dataset, batch_size=batch_size, shuffle=False)
fake_test_loader = DataLoader(fake_test_dataset, batch_size=batch_size, shuffle=False)

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, and optimizer
model = UNetVAE().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.5)

# Train the model
best_loss = float('inf')
for epoch in tqdm(range(num_epochs), desc="Training"):
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(train_loader, desc="Training Batches", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs, mean, log_var = model(inputs)

        # Compute loss
        reconstruction_loss = criterion(outputs, targets)
        kl_divergence = -0.6 * torch.mean(1 + log_var - mean.pow(2) - torch.exp(log_var))

        # L1 regularization
        l1_loss = 0
        for param in model.parameters():
            if param.requires_grad:
                l1_loss += torch.sum(torch.abs(param))

        loss = reconstruction_loss + kl_divergence + l1_lambda * l1_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save the model
    save_path = os.path.join("models", f"unet_vae_FF++_FS_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)

    # Update the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_save_path = os.path.join("models", "best_unet_vae_FF++_FS.pth")
        torch.save(model.state_dict(), best_save_path)
        print(f"Best model updated at epoch {epoch+1} with loss {best_loss:.4f}")

# Evaluate the model on test data
model.eval()
real_test_loss = 0
fake_test_loss = 0
real_test_losses = []
fake_test_losses = []

with torch.no_grad():
    for inputs, targets in tqdm(real_test_loader, desc="Testing Real Batches"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, mean, log_var = model(inputs)
        loss = criterion(outputs, targets)
        real_test_loss += loss.item()
        real_test_losses.append(loss.item())

    for inputs, targets in tqdm(fake_test_loader, desc="Testing Fake Batches"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, mean, log_var = model(inputs)
        loss = criterion(outputs, targets)
        fake_test_loss += loss.item()
        fake_test_losses.append(loss.item())

import os

# Define the path to the directory where you want to write the file
directory_path = "results"

# Create the directory if it does not exist
os.makedirs(directory_path, exist_ok=True)

with open(os.path.join(directory_path, "test_losses.txt"), "a+") as f:
    if f.tell() == 0:
        f.write("Real Test Loss,Fake Test Loss\n")
    f.write(f"{real_test_losses:.4f},{fake_test_losses:.4f}\n")
real_test_loss /= len(real_test_loader)
fake_test_loss /= len(fake_test_loader)

print(f"Real Test Loss: {real_test_loss:.4f}")
print(f"Fake Test Loss: {fake_test_loss:.4f}")

# Plot the distribution of losses for real and fake test datasets
plt.hist(real_test_losses, alpha=0.5, label='Real', density=True)
plt.hist(fake_test_losses, alpha=0.5, label='Fake', density=True)

kde_real = gaussian_kde(real_test_losses)
kde_fake = gaussian_kde(fake_test_losses)
x = np.linspace(min(min(real_test_losses), min(fake_test_losses)), max(max(real_test_losses), max(fake_test_losses)), 100)
plt.plot(x, kde_real(x), label='Real KDE')
plt.plot(x, kde_fake(x), label='Fake KDE')

plt.legend()
plt.xlabel('Loss')
plt.ylabel('Density')
plt.title('Distribution of Losses for Real and Fake Test Datasets')
plt.show()