import torch
from variational_autoencoder import VAE, load_dataset
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pickle

# Load model
vae = VAE(latent_dim=16)
vae.load_state_dict(torch.load("saved_models/model.pth"))
dataset = load_dataset()
# Generate reconstructions
num_images = 50
image_pairs = []
save_object = {"interpolation_path":[], "interpolation_images":[]}

# Make interpolation path
image_a, image_b = dataset[0][0], dataset[1][0]
image_a = image_a.view(32*32)
image_b = image_b.view(32*32)
z_a, _, _, _ = vae.forward(image_a)
z_a = z_a.detach().cpu().numpy()
z_b, _, _, _ = vae.forward(image_b)
z_b = z_b.detach().cpu().numpy()
interpolation_path = np.linspace(z_a, z_b, num=num_images)
# interpolation_path[:, 4] = np.linspace(-3, 3, num=num_images)
save_object["interpolation_path"] = interpolation_path

for i in range(num_images):
    # Generate 
    z = torch.Tensor(interpolation_path[i]).unsqueeze(0)
    gen_image = vae.decode(z).detach().numpy()
    gen_image = np.reshape(gen_image, (32, 32)) * 255
    save_object["interpolation_images"].append(gen_image)

fig, axs = plt.subplots(num_images, 1, figsize=(1, num_images))
image_pairs = []
for i in range(num_images):
    recon_image = save_object["interpolation_images"][i]
    # Add to plot
    axs[i].imshow(recon_image)

# Perform intrpolations
with open("interpolations.pkl", "wb") as f:
    pickle.dump(save_object, f)

plt.show()