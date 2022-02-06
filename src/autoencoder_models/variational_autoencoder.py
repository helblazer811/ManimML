import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()
  
# Download the MNIST Dataset
dataset = datasets.MNIST(root = "./data",
                         train = True,
                         download = True,
                         transform = tensor_transform)
  
# DataLoader is used to load the dataset 
# for training
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = 32,
                                     shuffle = True)
# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class VAE(torch.nn.Module):
    def __init__(self, latent_dim=5):
        super().__init__()
        self.latent_dim = latent_dim
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
        )
        self.mean_embedding = torch.nn.Linear(18, self.latent_dim)
        self.logvar_embedding = torch.nn.Linear(18, self.latent_dim)

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )

    def decode(self, z):
        return self.decoder(z)
  
    def forward(self, x):
        encoded = self.encoder(x)
        mean = self.mean_embedding(encoded)
        logvar = self.logvar_embedding(encoded)
        batch_size = x.shape[0]
        eps = torch.randn(batch_size, self.latent_dim)
        z = mean + torch.exp(logvar / 2) * eps
        reconstructed = self.decoder(z)
        return mean, logvar, reconstructed, x

def train_model():
    # Model Initialization
    model = VAE(latent_dim=16)
    # Validation using MSE Loss function
    def loss_function(mean, log_var, reconstructed, original, kl_beta=0.001):
        kl = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
        recon = torch.nn.functional.mse_loss(reconstructed, original)
        # print(f"KL Error {kl}, Recon Error {recon}")
        return kl_beta * kl + recon

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = 1e-3,
                                weight_decay = 1e-8)

    epochs = 100
    outputs = []
    losses = []
    for epoch in tqdm(range(epochs)):
        for (image, _) in loader:
            # Reshaping the image to (-1, 784)
            image = image.reshape(-1, 28*28)
            # Output of Autoencoder
            mean, log_var, reconstructed, image = model(image)
            # Calculating the loss function
            loss = loss_function(mean, log_var, reconstructed, image)
            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Storing the losses in a list for plotting
            if torch.isnan(loss):
                raise Exception()
            losses.append(loss.detach().cpu())
            outputs.append((epochs, image, reconstructed))

    torch.save(model.state_dict(), "saved_models/model.pth")

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    # Plotting the last 100 values
    plt.plot(losses)
    plt.show()

if __name__ == "__main__":
    train_model()