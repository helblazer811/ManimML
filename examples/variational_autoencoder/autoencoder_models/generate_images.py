import torch
from variational_autoencoder import VAE
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pickle

# Load model
vae = VAE(latent_dim=16)
vae.load_state_dict(torch.load("saved_models/model.pth"))
# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()
# Download the MNIST Dataset
dataset = datasets.MNIST(root = "./data",
                         train = True,
                         download = True,
                         transform = tensor_transform)
# Generate reconstructions
num_recons = 10
fig, axs = plt.subplots(num_recons, 2, figsize=(2, num_recons))
image_pairs = []
for i in range(num_recons):
    base_image, _ = dataset[i]
    base_image = base_image.reshape(-1, 28*28)
    _, _, recon_image, _ = vae.forward(base_image)
    base_image = base_image.detach().numpy()
    base_image = np.reshape(base_image, (28, 28)) * 255
    recon_image = recon_image.detach().numpy()
    recon_image = np.reshape(recon_image, (28, 28)) * 255
    # Add to plot
    axs[i][0].imshow(base_image)
    axs[i][1].imshow(recon_image)
    # image pairs
    image_pairs.append((base_image, recon_image))

with open("image_pairs.pkl", "wb") as f:
    pickle.dump(image_pairs, f)

plt.show()
