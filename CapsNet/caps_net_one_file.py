# Imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
# Load Data MNIST
use_cuda = False#torch.cuda.is_available()
print('USE CUDA:',use_cuda)
LAMBDA_RECON = 0.0005


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image



# (Set Random seed)
torch.manual_seed(100)

s_j1 = torch.FloatTensor([1, 1, 1, 1])
s_j0 = torch.FloatTensor([0, 0, 0, 0])
s_j2 = torch.FloatTensor([5, 3, 6, 2])

def squash(s_j_=[], epsilon=1e-7, dim = -1):
    norm = torch.norm(s_j_,p=2,dim=dim,keepdim=True)
    squared_norm = norm**2
    safe_norm = norm + epsilon
    squared_norm_by_1_plus_squared_norm = squared_norm / (1 + squared_norm)
    s_j_by_squared_norm = s_j_ / safe_norm
    s_j_squashed = squared_norm_by_1_plus_squared_norm * s_j_by_squared_norm
    #print("s_j", s_j_)
    # print("safe_norm",safe_norm)
    #print ("s_j_by_squared_norm", s_j_by_squared_norm)
    #print ("squared_norm_by_1_plus_squared_norm",squared_norm_by_1_plus_squared_norm)
    #print ("s_j_squashed", s_j_squashed)
    return s_j_squashed

'''
def squash2(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs

'''
def plot_squash():
    squashed_vector = torch.Tensor([])
    max = 100
    for i in range(0, 5):
        for j in range(0, max):
            v = torch.Tensor([j])/max + i  # t.rand(1) *40
            #print(v)
            v_squashed = squash2(v)
            squashed_vector = torch.cat((squashed_vector, v_squashed), 0)
        #sorted_vector, indices = t.sort(squashed_vector)
    # print(sorted_vector)
    #test = t.Tensor([1,2,3,4,5,6],[1,2,3,4,5,6])
    plt.plot(squashed_vector.numpy())
    plt.show()

#plot_squash()

class Primary_capsule(nn.Module):
    def __init__(self, in_dimension, out_dimension,capsule_dimensions, kernel_size,  stride=2, padding=0):
        super(Primary_capsule, self).__init__()
        self.capsule_dimensions = capsule_dimensions
        # define all the components that will be used in the NN (these can be reused)
        self.conv = nn.Conv2d(in_dimension, out_dimension, kernel_size=kernel_size, stride = stride, padding=padding)

    def forward(self, x):
        # define the acutal network
        in_size = x.size(0)  # get the batch size
        # chain function together to form the layers
        #print('x_size prime', x.size())
        x = self.conv(x)
        #print('x_size prime conv', x.size())
        x = squash(x)
        #print('x_size prime squashed', x.size())
        x = x.view(in_size, -1,self.capsule_dimensions)
        #print('x_size prime flattened', x.size())
        #print (x)
        #output is a flat array of squashed 8d vectors!
        return x
      
class Digit_capsule(nn.Module):
    def __init__(self):
        super(Digit_capsule, self).__init__()
        batch_size = 13
        self.num_caps_in = 6*6*32 # Number of capsules from the Previous layer
        self.num_caps_out = 10 # Number of capsules of the Output
        self.input_dimensions = 8 # Number of dimensions of the input capsule vectors
        self.output_dimensions = 16 # Number of dimensions of the output capsule vectors
        self.routings = 3 #number of routings
        # how to init weight matrix?
        self.weight_matrix = nn.Parameter(0.01 * torch.randn(self.num_caps_out, self.num_caps_in, self.output_dimensions, self.input_dimensions))
        self.u_hat = torch.zeros(self.num_caps_in, self.num_caps_out, self.output_dimensions)
        # how to init digit capsules?
        self.digit_capsules = torch.randn(self.num_caps_out, self.output_dimensions)
        

    def forward(self, x):
        # matmul input with weight_matrices to get u_hat
        #print ('x_size digit',x.size())
        #print ('weight_matrix_size', self.weight_matrix.size())
        u_hat = torch.squeeze(torch.matmul(self.weight_matrix, x[:, None, :, :, None]), dim=-1)
        #print('u_hat size', u_hat.size())
        routing_weights = Variable(torch.zeros(x.size(0), self.num_caps_out, self.num_caps_in))
        if use_cuda:
            routing_weights = routing_weights.cuda()
        #print('routing_weights size', routing_weights.size())
        u_hat_detached = u_hat.detach()
        # route u_hat to digit_capsules using dynamic routing by agreement

        assert self.routings > 0
        for i in range(self.routings):
            #coupling coefficients
            c = F.softmax(routing_weights, dim=1)
            #print('coefficients size', c.size())
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * u_hat, dim=-2, keepdim=True))
                
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                routing_weights = routing_weights + torch.sum(outputs * u_hat_detached, dim=-1)
        return torch.squeeze(outputs, dim=-2)


#Decoder ("Now let's build the decoder. It's quite simple: two dense (fully connected) ReLU layers followed by a dense output sigmoid layer")
class Decoder(nn.Module):
    def __init__(self, in_dim=16, out_dim=784):
        super(Decoder, self).__init__()
        HIDDEN_LAYER_FEATURES = [512, 1024]
        # define all the components that will be used in the NN (these can be reused)
        self.fc1 = nn.Linear(in_dim, HIDDEN_LAYER_FEATURES[0])
        self.fc2 = nn.Linear(HIDDEN_LAYER_FEATURES[0], HIDDEN_LAYER_FEATURES[1])
        self.fc3 = nn.Linear(HIDDEN_LAYER_FEATURES[1], out_dim)

    def forward(self, x):
        # chain function together to form the layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
      

# How to load MNIST: https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558
root = '../data'
download = True  # download MNIST dataset or not

trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans,
                       download=download)

test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 64

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

class Capsule_network(nn.Module):
    def __init__(self, batch_size):
        super(Capsule_network, self).__init__()
        self.conv = nn.Conv2d(1, 256, kernel_size=9, padding=(0,0))
        self.primary_capsules = Primary_capsule(
            in_dimension=256, out_dimension=256, kernel_size=9, capsule_dimensions=8)
        self.digit_capsules=Digit_capsule()#self.digit_capsules=DenseCapsule(6*6*32,8,10,16)
        #self.decoder = Decoder()

        self.decoder = nn.Sequential(
            nn.Linear(16*10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28*28*1),
            nn.Sigmoid()
        )

    def forward(self, x, y = None):
        # chain function together to form the layers
        x=self.conv(x)
        x=self.primary_capsules(x)
        x=self.digit_capsules(x)
        length = x.norm(dim = -1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.)
            if use_cuda:
                y = y.cuda()
            y = Variable(y)

        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction


# Margin Loss
def margin_loss(y_pred, y_true):
    m_plus = 0.9
    m_minus = 0.2
    lambda_down_weight =1# 0.5
    #print('y_pred', y_pred)
    #print('test',torch.clamp(m_plus - y_pred,min=0.))
    loss = y_true * torch.clamp(m_plus - y_pred,min=0.) ** 2 \
        + lambda_down_weight * (1- y_true) * torch.clamp(y_pred - m_minus,min=0.) ** 2
    margin_loss = loss.sum(dim=1).mean()
    #print ('Loss:',loss)
    return margin_loss


def reconstruction_loss(x, x_recon):
    recon_loss = nn.MSELoss()(x_recon, x.view(x.size(0),-1))
    return recon_loss

def capsule_loss(y_pred, y_true, x, x_recon, lambda_recon = 0.0005):
    m_loss = margin_loss(y_pred,y_true)
    r_loss = reconstruction_loss(x,x_recon)
    #print("m_loss", m_loss)
    #print("r_loss", r_loss)
    caps_loss = m_loss + lambda_recon * r_loss
    return caps_loss

  
def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()
    L_recon = nn.MSELoss()(x_recon, x)
    return L_margin + lam_recon * L_recon

def train_decoder_as_autoencoder(epoch, network):
    optimizer = Adam(network.parameters(), lr=0.1)
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

def show_reconstruction(model, test_loader, n_images):
    model.eval()
    for x, _ in test_loader:
        x_cuda =x[:min(n_images, x.size(0))]
        if use_cuda:
          x_cuda.cuda()
        x_cuda = Variable(x_cuda, volatile=True)
        _, x_recon = model(x_cuda)
        x_recon = x_recon.view(-1,1,28,28)
        print('x_cuda',x_cuda.size())
        print('x_recon',x_recon.size())
        data = np.concatenate([x_cuda.data, x_recon.data])
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        image = img * 255
        Image.fromarray(image.astype(np.uint8)).save('reconstructions' + "/real_and_recon.png")
        print()
        print('Reconstructed images are saved to %s/real_and_recon.png' % 'reconstructions')
        print('-' * 70)
        plt.imshow(plt.imread('reconstructions' + "/real_and_recon.png", ))
        plt.show()
        break
        
        
        
        
def train(epoch, network):
    optimizer = Adam(network.parameters(), lr=0.001)
    network.train()
    for j in range(0, epoch):
        for index, (x, target) in enumerate(train_loader):
            target_onehot = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 1.)
            if use_cuda:
              x= x.cuda()
              target_onehot = target_onehot.cuda()
            x, target = Variable(x), Variable(target_onehot)
            optimizer.zero_grad()
            # print('target size', target.size())
            #print('x_size train', x.size())
            output, reconstruction = network(x, target)
            
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
                
                
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for index, (x, y) in enumerate(test_loader):
        y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
        if use_cuda:
          x.cuda()
          y.cuda
        x, y = Variable(x, volatile=True), Variable(y)
        y_pred, x_recon = model(x)
        test_loss += caps_loss(y, y_pred, x, x_recon, LAMBDA_RECON).data[0] * x.size(0)  # sum up batch loss
        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        correct += y_pred.eq(y_true).cpu().sum()
        if index % 10 == 0:
                print('Test Epoch: {}'.format(index * len(x)))
    test_loss /= len(test_loader.dataset)
    return test_loss, correct / len(test_loader.dataset)





# plot_MNIST()
decoder_net=Decoder(784)
# train_decoder_as_autoencoder(1, decoder_net)
# plot_squash()

# routing_by_agreement('a')
capsule_network=Capsule_network(batch_size)
if use_cuda:
  capsule_network.cuda()
print('Start!')
#train(1, capsule_network)
#test_loss, test_acc = test(model=capsule_network, test_loader=test_loader)
#print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))
show_reconstruction(capsule_network, test_loader, 15)