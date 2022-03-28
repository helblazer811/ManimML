import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

"""
    These are utility functions that help to calculate the input and output
    sizes of convolutional neural networks
"""

def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)

def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    
    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
    
    return h, w

def convtransp2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0):
    h_w, kernel_size, stride, pad, dilation, out_pad = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation), num2tuple(out_pad)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    
    h = (h_w[0] - 1)*stride[0] - sum(pad[0]) + dialation[0]*(kernel_size[0]-1) + out_pad[0] + 1
    w = (h_w[1] - 1)*stride[1] - sum(pad[1]) + dialation[1]*(kernel_size[1]-1) + out_pad[1] + 1
    
    return h, w

def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    h_w_in, h_w_out, kernel_size, stride, dilation = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation)
    
    p_h = ((h_w_out[0] - 1)*stride[0] - h_w_in[0] + dilation[0]*(kernel_size[0]-1) + 1)
    p_w = ((h_w_out[1] - 1)*stride[1] - h_w_in[1] + dilation[1]*(kernel_size[1]-1) + 1)
    
    return (math.floor(p_h/2), math.ceil(p_h/2)), (math.floor(p_w/2), math.ceil(p_w/2))

def convtransp2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1, out_pad=0):
    h_w_in, h_w_out, kernel_size, stride, dilation, out_pad = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation), num2tuple(out_pad)
        
    p_h = -(h_w_out[0] - 1 - out_pad[0] - dilation[0]*(kernel_size[0]-1) - (h_w_in[0] - 1)*stride[0]) / 2
    p_w = -(h_w_out[1] - 1 - out_pad[1] - dilation[1]*(kernel_size[1]-1) - (h_w_in[1] - 1)*stride[1]) / 2
    
    return (math.floor(p_h/2), math.ceil(p_h/2)), (math.floor(p_w/2), math.ceil(p_w/2))

def load_dataset(train=True, digit=None):
    # Transforms images to a PyTorch Tensor
    tensor_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])
    
    # Download the MNIST Dataset
    dataset = datasets.MNIST(root = "./data",
                            train = train,
                            download = True,
                            transform = tensor_transform)
    # Load specific image
    if not digit is None:
        idx = dataset.train_labels == digit
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

    return dataset

def load_vae_from_path(path, latent_dim):
    model = VAE(latent_dim)
    model.load_state_dict(torch.load(path))
    
    return model

# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class VAE(torch.nn.Module):
    def __init__(self, latent_dim=5, layer_count=4, channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_shape = 32
        self.layer_count = layer_count
        self.channels = channels
        self.d = 128
        mul = 1
        inputs = self.channels
        out_sizes = [(self.in_shape, self.in_shape)]
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, self.d * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(self.d * mul))
            h_w = (out_sizes[-1][-1], out_sizes[-1][-1])
            out_sizes.append(conv2d_output_shape(h_w, kernel_size=4, stride=2, pad=1, dilation=1))
            inputs = self.d * mul
            mul *= 2

        self.d_max = inputs
        self.last_size = out_sizes[-1][-1]
        self.num_linear = self.last_size ** 2 * self.d_max
        # Encoder linear layers
        self.encoder_mean_linear = nn.Linear(self.num_linear, self.latent_dim)
        self.encoder_logvar_linear = nn.Linear(self.num_linear, self.latent_dim)
        # Decoder linear layer
        self.decoder_linear = nn.Linear(self.latent_dim, self.num_linear)

        mul = inputs // self.d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, self.d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(self.d * mul))
            inputs = self.d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, self.channels, 4, 2, 1))

    def encode(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        batch_size = x.shape[0]

        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))

        x = x.view(batch_size, -1)

        mean = self.encoder_mean_linear(x)
        logvar = self.encoder_logvar_linear(x)

        return mean, logvar

    def decode(self, x):
        x = x.view(x.shape[0], self.latent_dim)
        x = self.decoder_linear(x)
        x = x.view(x.shape[0], self.d_max, self.last_size, self.last_size)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)
        x = getattr(self, "deconv%d" % (self.layer_count + 1))(x)
        x = torch.sigmoid(x)
        return x
  
    def forward(self, x):
        batch_size = x.shape[0]
        mean, logvar = self.encode(x)
        eps = torch.randn(batch_size, self.latent_dim)
        z = mean + torch.exp(logvar / 2) * eps
        reconstructed = self.decode(z)
        return mean, logvar, reconstructed, x

def train_model(latent_dim=16, plot=True, digit=1, epochs=200):
    dataset = load_dataset(train=True, digit=digit)
    # DataLoader is used to load the dataset 
    # for training
    loader = torch.utils.data.DataLoader(dataset = dataset,
                                        batch_size = 32,
                                        shuffle = True)
    # Model Initialization
    model = VAE(latent_dim=latent_dim)
    # Validation using MSE Loss function
    def loss_function(mean, log_var, reconstructed, original, kl_beta=0.0001):
        kl = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
        recon = torch.nn.functional.mse_loss(reconstructed, original)
        # print(f"KL Error {kl}, Recon Error {recon}")
        return kl_beta * kl + recon

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = 1e-4,
                                weight_decay = 0e-8)

    outputs = []
    losses = []
    for epoch in tqdm(range(epochs)):
        for (image, _) in loader:
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

    torch.save(model.state_dict(), os.path.join(os.environ["PROJECT_ROOT"], f"saved_models/model_dim{latent_dim}.pth"))

    if plot:
        # Defining the Plot Style
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        
        # Plotting the last 100 values
        plt.plot(losses)
        plt.show()

if __name__ == "__main__":
    train_model(latent_dim=2, digit=2, epochs=40)
