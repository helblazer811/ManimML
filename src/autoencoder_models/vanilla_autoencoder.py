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
    def __init__(self):
        super().__init__()
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
        self.mean_embedding = torch.nn.Linear(18, 9)
        self.logvar_embedding = torch.nn.Linear(18, 9)

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
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
  
    def forward(self, x):
        encoded = self.encoder(x)
        mean = self.mean_embedding(encoded)
        logvar = self.logvar_embedding(encoded)
        combined = torch.cat((mean, logvar), dim=1)
        reconstructed = self.decoder(combined)
        return mean, logvar, reconstructed, x

# Model Initialization
model = VAE()
# Validation using MSE Loss function
def loss_function(mean, log_var, reconstructed, original):
    kl = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
    recon = torch.nn.functional.mse_loss(reconstructed, original)

    return kl + recon

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)

epochs = 10
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
      losses.append(loss.detach().cpu())
    outputs.append((epochs, image, reconstructed))

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
  
# Plotting the last 100 values
print(losses)
plt.plot(losses[-100:])
plt.show()