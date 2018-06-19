# Imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from custom_layers import Primary_capsule, Decoder, Digit_capsule, squash
# Load Data MNIST
use_cuda = torch.cuda.is_available()

LAMBDA_RECON = 0.0005

# How to load MNIST: https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558
root = '../data'
download = False  # download MNIST dataset or not

trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans,
                       download=download)

test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 13

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

print('==>>> total training batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))


def plot_MNIST():
    # Plot Data (= Show Images)
    n_samples = 8

    plt.figure(figsize=(n_samples * 2, 3))
    for index, (x, target) in enumerate(train_loader):
        if(index == n_samples):
            break
        plt.subplot(1, n_samples+1, index+1)
        # torch.randn(28,28)
        #print('Target:', target[index])

        sample_image = x[index].view(28, 28)
        # print(sample_image)
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")

    plt.show()


# (Set Random seed)
torch.manual_seed(100)


# Capsule Dimensions
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
caps1_n_dims = 8


# Margin Loss
def margin_loss(y_pred, y_true):
    m_plus = 0.9
    m_minus = 0.1
    lambda_down_weight = 0.5
    #print('output', y_pred.size())
    #print('y_pred', y_pred)
    loss = y_true * torch.clamp(m_plus - y_pred,min=0.) ** 2 \
        + lambda_down_weight * (1- y_true) * torch.clamp(y_pred - m_minus,min=0.) ** 2
    margin_loss = loss.sum(dim=1).mean()
    #print ('Loss:',margin_loss)
    return margin_loss 


def reconstruction_loss(x, x_recon):
    recon_loss = nn.MSELoss()(x_recon, x)
    return recon_loss

def capsule_loss(y_pred, y_true, x, x_recon, lambda_recon = 0.0005):
    caps_loss = margin_loss(y_pred,y_true) + lambda_recon * reconstruction_loss(x,x_recon)
    return caps_loss


def train_decoder_as_autoencoder(epoch, network):
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
    network.train()
    for j in range(0, epoch):
        for index, (x, target) in enumerate(train_loader):
            x = torch.squeeze(x)
            x = x.view(x.size(0), -1)

            x, target = Variable(x), Variable(target)
            optimizer.zero_grad()
            # print('target size', target.size())
            #print('x size', x.size())
            output = network(x)
            #print('output size', output.size())
            # x = x.long()
            # loss =F.nll_loss(output, x) #margin_loss(output,target)
            loss = F.mse_loss(output, x)
            # margin_loss.backward()
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    j+1, index * len(x), len(train_loader.dataset),
                    100. * index / len(train_loader), loss.data[0]))


def train(epoch, network):
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
    network.train()
    for j in range(0, epoch):
        for index, (x, target) in enumerate(train_loader):
            target_onehot = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 1.)
            x, target = Variable(x), Variable(target_onehot)
            optimizer.zero_grad()
            # print('target size', target.size())
            #print('x_size train', x.size())
            output, reconstruction = network(x)
            #print('output size',  output.size())
            # loss =F.nll_loss(output, x) #margin_loss(output,target)
            loss = capsule_loss(output, target, x, reconstruction, LAMBDA_RECON)
            # margin_loss.backward()
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    j+1, index * len(x), len(train_loader.dataset),
                    100. * index / len(train_loader), loss.data[0]))


class Capsule_network(nn.Module):
    def __init__(self, batch_size):
        super(Capsule_network, self).__init__()
        self.conv = nn.Conv2d(1, 256, kernel_size=9, padding=(0,0))
        self.primary_capsules = Primary_capsule(
            in_dimension=256, out_dimension=256, kernel_size=9, capsule_dimensions=8)
        self.digit_capsules=Digit_capsule()
        self.decoder = Decoder()

    def forward(self, x, y = None):
        
        
        # chain function together to form the layers
        x=self.conv(x)
        x=self.primary_capsules(x)
        x=self.digit_capsules(x)
        length = x.norm(dim = -1)

        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
       
        return length, reconstruction


# plot_MNIST()
decoder_net=Decoder(784)
# train_decoder_as_autoencoder(1, decoder_net)
# plot_squash()

# routing_by_agreement('a')
capsule_network=Capsule_network(batch_size)
train(1, capsule_network)
