import pickle
import sys
import os
sys.path.append(os.environ["PROJECT_ROOT"])
from autoencoder_models.variational_autoencoder import VAE, load_dataset, load_vae_from_path
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy
import scipy.stats
import cv2

def binned_images(model_path, num_x_bins=6, plot=False):
    latent_dim = 2
    model = load_vae_from_path(model_path, latent_dim)
    image_dataset = load_dataset(digit=2)
    # Compute embedding
    num_images = 500
    embedding = []
    images = []
    for i in range(num_images):
        image, _ = image_dataset[i]
        mean, _, recon, _ = model.forward(image)
        mean = mean.detach().numpy()
        recon = recon.detach().numpy()
        recon = recon.reshape(32, 32)
        images.append(recon.squeeze())
        if latent_dim > 2:
            mean = mean[:2]
        embedding.append(mean)
    images = np.stack(images)
    tsne_points = np.array(embedding)
    tsne_points = (tsne_points - tsne_points.mean(axis=0))/(tsne_points.std(axis=0))
    # make vis 
    num_points = np.shape(tsne_points)[0]
    x_min = np.amin(tsne_points.T[0])
    y_min = np.amin(tsne_points.T[1])
    y_max = np.amax(tsne_points.T[1])
    x_max = np.amax(tsne_points.T[0])
    # make the bins from the ranges
    # to keep it square the same width is used for x and y dim
    x_bins, step = np.linspace(x_min, x_max, num_x_bins, retstep=True)
    x_bins = x_bins.astype(float)
    num_y_bins = np.absolute(np.ceil((y_max - y_min)/step)).astype(int)
    y_bins = np.linspace(y_min, y_max, num_y_bins)
    # sort the tsne_points into a 2d histogram
    tsne_points = tsne_points.squeeze()
    hist_obj = scipy.stats.binned_statistic_dd(tsne_points, np.arange(num_points), statistic='count', bins=[x_bins, y_bins], expand_binnumbers=True)
    # sample one point from each bucket
    binnumbers = hist_obj.binnumber
    num_x_bins = np.amax(binnumbers[0]) + 1
    num_y_bins = np.amax(binnumbers[1]) + 1
    binnumbers = binnumbers.T
    # some places have no value in a region
    used_mask = np.zeros((num_y_bins, num_x_bins))
    image_bins = np.zeros((num_y_bins, num_x_bins, 3, np.shape(images)[2],  np.shape(images)[2]))
    for i, bin_num in enumerate(list(binnumbers)):
        used_mask[bin_num[1], bin_num[0]] = 1
        image_bins[bin_num[1], bin_num[0]] = images[i]
    # plot a grid of the images
    fig, axs = plt.subplots(nrows=np.shape(y_bins)[0], ncols=np.shape(x_bins)[0], constrained_layout=False, dpi=50)
    images = []
    bin_indices = []
    for y in range(num_y_bins):
        for x in range(num_x_bins):
            if used_mask[y, x] > 0.0:
                image = np.uint8(image_bins[y][x].squeeze()*255)
                image = np.rollaxis(image, 0, 3)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axs[num_y_bins - 1 - y][x].imshow(image)
                images.append(image)
                bin_indices.append((y, x))
            axs[y, x].axis('off')
    if plot:
        plt.axis('off')
        plt.show()
    else:
        return images, bin_indices

def generate_disentanglement(model_path="saved_models/model_dim2.pth"):
    """Generates disentanglement visualization and serializes it"""
    # Disentanglement object
    disentanglement_object = {}
    # Make Disentanglement
    images, bin_indices = binned_images(model_path)
    disentanglement_object["images"] = images
    disentanglement_object["bin_indices"] = bin_indices
    # Serialize Images
    with open("disentanglement.pkl", "wb") as f:
        pickle.dump(disentanglement_object, f)

if __name__ == "__main__":
    plot = False
    if plot:
        model_path = "saved_models/model_dim2.pth"
        #uniform_image_sample(model_path)
        binned_images(model_path)
    else:
        generate_disentanglement()