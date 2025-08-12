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



# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the trained autoencoder model
model = UNetVAE().to(device)
model.load_state_dict(torch.load('DeepFake/models/best_unet_vae.pth', map_location=device))

# Define the loss function
def loss_function(recon_x, x, mean, log_var):
    BCE = nn.MSELoss()(recon_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD, BCE, KLD

# Create dataset and data loader

real_val_path = 'DeepFake/DFDCP/validation/real'
fake_val_path = 'DeepFake/DFDCP/validation/fake'
real_test_path = 'DeepFake/DFDCP/test/real'
fake_test_path = 'DeepFake/DFDCP/test/fake'
batch_size = 30

real_val_dataset = ImageDataset(real_val_path)
fake_val_dataset = ImageDataset(fake_val_path)
real_test_dataset = ImageDataset(real_test_path)
fake_test_dataset = ImageDataset(fake_test_path)
real_val_loader = DataLoader(real_val_dataset, batch_size=batch_size, shuffle=False)
fake_val_loader = DataLoader(fake_val_dataset, batch_size=batch_size, shuffle=False)
real_test_loader = DataLoader(real_test_dataset, batch_size=batch_size, shuffle=False)
fake_test_loader = DataLoader(fake_test_dataset, batch_size=batch_size, shuffle=False)

# Calculate the threshold using the ROC curve and Youden's Index
model.eval()
original_losses = []
deepfake_losses = []
with torch.no_grad():
    for i, (inputs, targets) in enumerate(tqdm(real_val_loader, desc="Calculating original losses")):
        # Test on original images
        inputs = inputs.to(device)  # Move inputs to GPU
        targets = targets.to(device)  # Move targets to GPU
        reconstructed_images, mean, log_var = model(inputs)
        reconstruction_loss, _, _ = loss_function(reconstructed_images, targets, mean, log_var)
        original_losses.extend([reconstruction_loss.item()])

    for i, (inputs, targets) in enumerate(tqdm(fake_val_loader, desc="Calculating deepfake losses")):
        # Test on deepfake images
        inputs = inputs.to(device)  # Move inputs to GPU
        targets = targets.to(device)  # Move targets to GPU
        reconstructed_images, mean, log_var = model(inputs)
        reconstruction_loss, _, _ = loss_function(reconstructed_images, targets, mean, log_var)
        deepfake_losses.extend([reconstruction_loss.item()])

# # Plot the distribution of the reconstruction losses
# plt.figure(figsize=(10,6))
# plt.hist(original_losses, alpha=0.5, label='Original', density=True)
# plt.hist(deepfake_losses, alpha=0.5, label='Deepfake', density=True)
# plt.xlabel('Reconstruction Loss')
# plt.ylabel('Density')
# plt.title('Distribution of Reconstruction Loss')
# plt.legend()
# plt.show()

thresholds = np.linspace(min(min(original_losses), min(deepfake_losses)), max(max(original_losses), max(deepfake_losses)), 100)

tp_rates = []
fp_rates = []
for i, threshold in enumerate(tqdm(thresholds, desc="Calculating ROC curve")):
    tp = sum(1 for loss in original_losses if loss < threshold)
    fn = sum(1 for loss in original_losses if loss >= threshold)
    fp = sum(1 for loss in deepfake_losses if loss < threshold)
    tn = sum(1 for loss in deepfake_losses if loss >= threshold)
    tp_rate = tp / (tp + fn)
    fp_rate = fp / (fp + tn)
    tp_rates.append(tp_rate)
    fp_rates.append(fp_rate)

# # Plot the ROC curve
# plt.figure(figsize=(10,6))
# plt.plot(fp_rates, tp_rates, label='ROC Curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
# plt.show()

youden_indices = [tp_rate - fp_rate for tp_rate, fp_rate in zip(tp_rates, fp_rates)]
optimal_threshold_index = np.argmax(youden_indices)
optimal_threshold = thresholds[optimal_threshold_index]
# print(f'Optimal Threshold: {optimal_threshold}')

# Evaluate the UNetVAE model on the test sets
model.eval()
test_accuracies = []
for i, test_loader in enumerate([real_test_loader, fake_test_loader]):
    correct = 0
    total = 0
    for j, (inputs, targets) in enumerate(tqdm(test_loader, desc=f"Evaluating test set {i+1}")):
        # Test on images
        inputs = inputs.to(device)  # Move inputs to GPU
        targets = targets.to(device)  # Move targets to GPU
        reconstructed_images, mean, log_var = model(inputs)
        reconstruction_loss, _, _ = loss_function(reconstructed_images, targets, mean, log_var)
        if i == 0:  # Original images
            if reconstruction_loss.item() < optimal_threshold:
                correct += 1
        else:  # Deepfake images
            if reconstruction_loss.item() >= optimal_threshold:
                correct += 1
        total += 1
    accuracy = correct / total
    test_accuracies.append(accuracy)
    # print(f'Test Accuracy on {["Original", "Deepfake"][i]}: {accuracy}')

# Print the results
# print('Test Accuracies:')
# print(f'Original: {test_accuracies[0]}')
# print(f'Deepfake: {test_accuracies[1]}')

# Calculate the overall accuracy
overall_accuracy = (test_accuracies[0] + test_accuracies[1]) / 2
# print(f'Overall Accuracy: {overall_accuracy}')

# Calculate the predictions
y_pred = []
y_true = []
for i, test_loader in enumerate([real_test_loader, fake_test_loader]):
    for j, (inputs, targets) in enumerate(test_loader):
        # Test on images
        inputs = inputs.to(device)  # Move inputs to GPU
        targets = targets.to(device)  # Move targets to GPU
        reconstructed_images, mean, log_var = model(inputs)
        reconstruction_loss, _, _ = loss_function(reconstructed_images, targets, mean, log_var)
        if i == 0:  # Original images
            y_true.extend([0]*inputs.shape[0])
            y_pred.extend([1 if loss.item() < optimal_threshold else 0 for loss in [reconstruction_loss]*inputs.shape[0]])
        else:  # Deepfake images
            y_true.extend([1]*inputs.shape[0])
            y_pred.extend([1 if loss.item() >= optimal_threshold else 0 for loss in [reconstruction_loss]*inputs.shape[0]])

# Calculate the F1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='macro')
print(f'F1 Score: {f1}')

# Calculate the ROC AUC score
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_true, y_pred)
print(f'ROC AUC Score: {roc_auc}')

# Calculate the classification report
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

# Calculate the confusion matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)