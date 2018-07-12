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
from PIL import ImageFont
from PIL import ImageDraw 
import numpy as np
import math
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from load_sketches import SketchDataset, Rescale, ToTensor, Dilate, ToInt, Invert, ToNumpy,RandomRotation, RandomCrop, SKETCH_LIST, SKETCH_PATH, show_sketches, show_sketches
from torch.utils.data.sampler import SubsetRandomSampler
import time


#from skopt.plots import plot_convergence
#from skopt.plots import plot_objective, plot_evaluations
#from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
#from sklearn.model_selection import GridSearchCV
#from skorch.net import NeuralNetClassifier

#from sklearn.datasets import fetch_mldata
#mnist = fetch_mldata('MNIST original', data_home='custom_data_home')
import sys
# Load Data MNIST
use_cuda = torch.cuda.is_available()
print('USE CUDA:', use_cuda)



if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# (Set Random seed)
torch.manual_seed(4789)
img_size = 48
LAMBDA_RECON=0.0005*img_size*img_size
transforms_composed = transforms.Compose([
    RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(size=1111, padding=10),
    ToNumpy(),
    Invert(),
    Dilate(),
    Dilate(),
    Dilate(),
    Dilate(),
    Dilate(),
    Dilate(),
    Dilate(),
    Dilate(),
    Dilate(),
    Dilate(),
    Dilate(),
    Rescale(img_size),
    ToInt(),
    ToTensor()])#, Normalize()])

    
sketch_dataset = SketchDataset(file_list = SKETCH_LIST, 
                                root_dir = SKETCH_PATH,
                                #categories = ['cat', 'dog', 'butterfly','wheel', 'wine-bottle','apple', 'book', 'moon', 'pig', 'ship'], 
                                #categories = ['butterfly','ship'], 
                                transform = transforms_composed
                                )

num_cateories = len(sketch_dataset.categories)
batch_size = 3
validation_split = .15
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(sketch_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(sketch_dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(sketch_dataset, batch_size=batch_size,
sampler=test_sampler)



print('==>>> total training batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))


def count_samples_per_class(train_loader_, test_loader_):
    train_sample_count = torch.zeros(10)
    for index, (x, target) in enumerate(train_loader_):
        for label in target:
            for class_index in range(0, 10):
                if label == class_index:
                    train_sample_count[class_index] += 1
    print('number of samples per class train', train_sample_count)

    test_sample_count = torch.zeros(10)
    for index, (x, target) in enumerate(test_loader_):
        for label in target:
            for class_index in range(0, 10):
                if label == class_index:
                    test_sample_count[class_index] += 1
    print('number of samples per class test', test_sample_count)
    total_sample_count = test_sample_count+train_sample_count
    print('number of samples per class total', total_sample_count)
    print('mean', total_sample_count.mean())
    print('std', total_sample_count.std())
    print('var', total_sample_count.var())


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

        sample_image = x[index].view(img_size, img_size)
        # print(sample_image)
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")

    plt.show()


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


s_j1 = torch.FloatTensor([1, 1, 1, 1])
s_j0 = torch.FloatTensor([0, 0, 0, 0])
s_j2 = torch.FloatTensor([5, 3, 6, 2])


def squash(s_j_=[], epsilon=1e-7, dim=-1):
    norm = torch.norm(s_j_, p=2, dim=dim, keepdim=True)
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
            # print(v)
            v_squashed = squash2(v)
            squashed_vector = torch.cat((squashed_vector, v_squashed), 0)
        #sorted_vector, indices = t.sort(squashed_vector)
    # print(sorted_vector)
    #test = t.Tensor([1,2,3,4,5,6],[1,2,3,4,5,6])
    plt.plot(squashed_vector.numpy())
    plt.show()

# plot_squash()


class Primary_capsule(nn.Module):
    def __init__(self, in_channels, out_channels, capsule_dimension, kernel_size,  stride=2, padding=0):
        super(Primary_capsule, self).__init__()
        self.capsule_dimension = capsule_dimension
        # define all the components that will be used in the NN (these can be reused)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # define the acutal network
        in_size = x.size(0)  # get the batch size
        # chain function together to form the layers
        #print('x_size prime', x.size())
        x = self.conv(x)
        #print('x_size prime conv', x.size())
        x = squash(x)
        #print('x_size prime squashed', x.size())
        x = x.view(in_size, -1, self.capsule_dimension)
        #print('x_size prime flattened', x.size())
        #print (x)
        # output is a flat array of squashed 8d vectors!
        return x


class Fully_connected_capsule(nn.Module):
    def __init__(self, in_dimension = 8, out_dimension = 16, num_caps_in = 100352, num_caps_out = 10, routings = 3):
        super(Fully_connected_capsule, self).__init__()
        self.num_caps_in = num_caps_in #100352#6*6*32  # Number of capsules from the Previous layer
        self.num_caps_out = num_caps_out#10  # Number of capsules of the Output
        self.in_dimension = in_dimension# 8  # Number of dimension of the input capsule vectors
        self.out_dimension = out_dimension  # Number of dimension of the output capsule vectors
        self.routings = routings  # number of routings
        # how to init weight matrix?
        self.weight_matrix = nn.Parameter(
            0.01 * torch.randn(self.num_caps_out, self.num_caps_in, self.out_dimension, self.in_dimension))
        self.u_hat = torch.zeros(
            self.num_caps_in, self.num_caps_out, self.out_dimension)
        # how to init digit capsules?
        self.Fully_connected_capsules = torch.randn(
            self.num_caps_out, self.out_dimension)

    def forward(self, x):
        # matmul input with weight_matrices to get u_hat
        #print ('x_size digit',x.size())
        #print ('weight_matrix_size', self.weight_matrix.size())
        u_hat = torch.squeeze(torch.matmul(
            self.weight_matrix, x[:, None, :, :, None]), dim=-1)
        #print('u_hat size', u_hat.size())
        routing_weights = Variable(torch.zeros(
            x.size(0), self.num_caps_out, self.num_caps_in))
        if use_cuda:
            routing_weights = routing_weights.cuda()
        #print('routing_weights size', routing_weights.size())
        u_hat_detached = u_hat.detach()
        # route u_hat to Fully_connected_capsules using dynamic routing by agreement

        assert self.routings > 0
        for i in range(self.routings):
            # coupling coefficients
            c = F.softmax(routing_weights, dim=1)
            #print('coefficients size', c.size())
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(
                    torch.sum(c[:, :, :, None] * u_hat, dim=-2, keepdim=True))

            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(
                    torch.sum(c[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                routing_weights = routing_weights + \
                    torch.sum(outputs * u_hat_detached, dim=-1)
        return torch.squeeze(outputs, dim=-2)


class Capsule_network(nn.Module):
    def __init__(self, routings):
        super(Capsule_network, self).__init__()
        self.final_out_dimension = 24
        kernel_size = 9
        padding = 0
        num_convs = 3
        primary_capsule_in_cannels = 256
        primary_capsule_out_cannels = 256
        primary_capsule_dimension = 16
        convoluted_img_size = int((img_size - (num_convs * (kernel_size - 1 + padding))) / 2)
        digit_caps_in = int(math.pow(convoluted_img_size,2)*(primary_capsule_out_cannels/primary_capsule_dimension))
        self.conv_1 = nn.Conv2d(1, 256, kernel_size=kernel_size, padding=(padding, padding))
        self.conv_2 = nn.Conv2d(256, primary_capsule_in_cannels, kernel_size=kernel_size, padding=(padding, padding))
        self.primary_capsules = Primary_capsule(
            in_channels=primary_capsule_in_cannels, out_channels=primary_capsule_out_cannels, kernel_size=kernel_size, capsule_dimension=primary_capsule_dimension)
        # self.Fully_connected_capsules=DenseCapsule(6*6*32,8,10,16)
        #self.Fully_connected_capsules_1 = Fully_connected_capsule(in_dimension = primary_capsule_dimension, out_dimension = 16, num_caps_in = digit_caps_in, num_caps_out = 64, routings = routings)
        #self.Fully_connected_capsules_2 = Fully_connected_capsule(in_dimension = 16, out_dimension = self.final_out_dimension, num_caps_in = 64, num_caps_out = 10, routings = routings)
        self.Fully_connected_capsules_1 = Fully_connected_capsule(in_dimension = primary_capsule_dimension, out_dimension = self.final_out_dimension, num_caps_in = digit_caps_in, num_caps_out = num_cateories, routings = routings)

        self.decoder = nn.Sequential(
            nn.Linear(self.final_out_dimension*num_cateories, 3072),
            nn.ReLU(inplace=True),
            nn.Linear(3072, 6144),
            nn.ReLU(inplace=True),
            nn.Linear(6144, img_size*img_size*1),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        # chain function together to form the layers
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.primary_capsules(x)
        x = self.Fully_connected_capsules_1(x)
        #x = self.Fully_connected_capsules_2(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            #print('length',length)
            #print('index', index.data)
            #print('index.view(-1, 1).data', index.view(-1, 1).data)
            # if use_cuda:
            #   index = index.cuda()
            y = torch.zeros(length.size()).scatter_(
                1, index.view(-1, 1).data, 1.)
            y = Variable(y)
        #print((x * y[:, :, None]).view(x.size(0), -1))
        #print(x * y[:, :, None]).view(x.size(0), -1)

        output_capsules_masked = (x * y[:, :, None]).view(x.size(0), -1)
        reconstruction = self.decoder(output_capsules_masked)
        return length, reconstruction


# Margin Loss
def margin_loss(y_pred, y_true):
    m_plus = 0.9
    m_minus = 0.1
    lambda_down_weight = 0.5  # 0.5
    #print('y_pred', y_pred)
    #print('test',torch.clamp(m_plus - y_pred,min=0.))
    loss = y_true * torch.clamp(m_plus - y_pred, min=0.) ** 2 \
        + lambda_down_weight * (1 - y_true) * \
        torch.clamp(y_pred - m_minus, min=0.) ** 2
    margin_loss = loss.sum(dim=1).mean()
    #print ('Loss:',loss)
    return margin_loss


def rmse(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y - y_hat).pow(2)))


def reconstruction_loss(x, x_recon):

    #recon_loss = nn.MSELoss()(x_recon, x)
    if False:
        plt.imshow(x_recon.detach().numpy(), cmap="binary")
        plt.show()
    x_reshaped = x.view(x.size(0), -1)
    x_recon_reshaped = x_recon.view(x_recon.size(0), img_size, img_size)
    # rmse loss ausprobieren!!!
    #recon_loss = nn.MSELoss()(x_recon, x_reshaped)
    recon_loss = rmse(x, x_recon_reshaped)
    return recon_loss


def capsule_loss(y_pred, y_true, x, x_recon, lambda_recon=0.0005):
    m_loss = margin_loss(y_pred, y_true)
    r_loss = reconstruction_loss(x, x_recon)
    #print("m_loss", m_loss)
    #print("r_loss", r_loss)
    caps_loss = m_loss + lambda_recon * r_loss
    return caps_loss, r_loss, m_loss


"""
def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """"""
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """"""
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()
    L_recon = nn.MSELoss()(x_recon, x)
    return L_margin + lam_recon * L_recon, L_recon, L_margin
"""


def train_decoder_as_autoencoder(epoch, network):
    optimizer = optim.Adam(network.parameters(), lr=0.1)
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

def show_reconstruction(model, test_loader, n_images, path, file_name):
    model.eval()
    if use_cuda:
        model = model.cuda()
    for sample in test_loader:
        x = sample['image']
        x = x.unsqueeze(1).float()
        y = sample['label_index']
        #print('x shape', x.shape)
        x = x[:min(n_images, x.size(0))]
        if use_cuda:
            x = x.cuda()
        # print('x_cuda',x_cuda.type())
        x = Variable(x)
        y_pred, x_recon = model(x)
        value, y_pred_label = y_pred.max(1)
        x_recon = x_recon.view(-1, 1, img_size, img_size)
        # print('x_cuda',x_cuda.type())
        # print('x_recon',x_recon.type())
        data = np.concatenate([x.data, x_recon.data])
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        img = img * 255
        img = Image.fromarray(img.astype(np.uint8))

        font = ImageFont.load_default()
        draw = ImageDraw.Draw(img)
        #print('y',y.numpy()[0])
        #print('y_pred',y_pred_label.data.cpu().numpy())
        #print('value', value)
        draw.text((0, 0),'R:{}'.format(y.numpy()),(255),font=font)
        draw.text((0, 8),'P:{}'.format( y_pred_label.data.cpu().numpy()),(255),font=font)
        file_full_path = '{0}/{1}.png'.format(path, file_name)
        img.save(file_full_path)
        print()
        print('Reconstructed images are saved to ' + file_full_path)
        print('-' * 70)
        #plt.imshow(plt.imread('reconstructions' + "/real_and_recon.png", ))
        # plt.imshow(image)
        # plt.show()
        break
def show_all_capsules(model, test_loader, path, file_name):
    model.eval()
    if use_cuda:
        model = model.cuda()
    for sample in test_loader:
        x = sample['image']
        x = x.unsqueeze(1).float()
        y = sample['label_index']
        y_onehot = torch.zeros(y.size(
                0), num_cateories).scatter_(1, y.view(-1, 1), 1.)
        #print('x shape', x.shape)
        x = x[:min(1, x.size(0))]
        if use_cuda:
            x = x.cuda()
        # print('x_cuda',x_cuda.type())
        x = Variable(x)
        y_pred_test, x_recon_test = model(x)

        value, y_pred_label = y_pred.max(1)
        x_recon = x_recon.view(-1, 1, img_size, img_size)
        # print('x_cuda',x_cuda.type())
        # print('x_recon',x_recon.type())
        data = np.concatenate([x.data, x_recon.data])
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        img = img * 255
        img = Image.fromarray(img.astype(np.uint8))

        font = ImageFont.load_default()
        draw = ImageDraw.Draw(img)
        #print('y',y.numpy()[0])
        #print('y_pred',y_pred_label.data.cpu().numpy())
        #print('value', value)
        draw.text((0, 0),'R:{}'.format(y.numpy()),(255),font=font)
        draw.text((0, 8),'P:{}'.format( y_pred_label.data.cpu().numpy()),(255),font=font)
        file_full_path = '{0}/{1}.png'.format(path, file_name)
        img.save(file_full_path)
        print()
        print('Reconstructed images are saved to ' + file_full_path)
        print('-' * 70)
        #plt.imshow(plt.imread('reconstructions' + "/real_and_recon.png", ))
        # plt.imshow(image)
        # plt.show()
        break

def show_batch(test_loader,path,file_name):
    for sample in test_loader:
        x = sample['image']
        x = x.unsqueeze(1).float()
        #print('x shape', x.shape)
        x = x[:min(x.shape[0], x.size(0))]
        x = Variable(x)
        data = x.data
        img = combine_images(np.transpose(data.numpy(), [0, 2, 3, 1]))
        image = img * 255
        
        #Image.fromarray(image.astype(np.uint8)).save(
        #    '{0}/{1}.png'.format(path, file_name))
        print()
        print('Sample images are saved to %s/real_and_recon.png' %
              'reconstructions')
        print('-' * 70)
        #plt.imshow(plt.imread('reconstructions' + "/real_and_recon.png", ))
        plt.imshow(image)
        plt.show()
        break

def train(epoch, network, learning_rate, lamda_recon, train_loader):
    once = True
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for j in range(0, epoch):
        network.train()
        for index, sample in enumerate(train_loader):
            x = sample['image']
            x = x.unsqueeze(1).float()
            target = sample['label_index']
            #print('label_shape: ', label)
            #print('x_shape: ', x.shape)
            #print('target_shape: ', target.shape)
            if use_cuda:
                x = x.cuda()
                target = target.cuda()
            target_onehot = torch.zeros(target.size(
                0), num_cateories).scatter_(1, target.view(-1, 1), 1.)
            #target_onehot = target_onehot.cuda()
            x, target = Variable(x), Variable(target_onehot)
            optimizer.zero_grad()
            # print('target size', target.size())
            #print('x_size train', x.size())
            output, reconstruction = network(x, target)

            #print('output size',  output.size())
            # loss =F.nll_loss(output, x) #margin_loss(output,target)
            loss, recon_loss, margin_loss = capsule_loss(
                output, target, x, reconstruction, lamda_recon)
            # margin_loss.backward()
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}| Reconstruction Loss: {:.6f} Margin Loss: {:.6f}'.format(
                    j+1, index * len(x), len(train_loader.dataset),
                    100. * index / len(train_loader), loss.data[0], recon_loss.data[0], margin_loss.data[0]))
        lr_decay.step()
        #test_loss, test_acc = test(model=capsule_network, test_loader=test_loader)
        #print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))
        #show_reconstruction(network, test_loader, 50)


def test(model, test_loader, lamda_recon):
    model.eval()
    test_loss = 0
    correct = 0
    for index, sample in enumerate(test_loader):
        x = sample['image']
        x = x.unsqueeze(1).float()
        target = sample['label_index']
        if use_cuda:
            x = x.cuda()
            target = target.cuda()
        target = torch.zeros(target.size(0), num_cateories).scatter_(1, target.view(-1, 1), 1.)
        x, target = Variable(x, volatile=True), Variable(target)
        y_pred, x_recon = model(x)
        loss, recon_loss, margin_loss = capsule_loss(
            target, y_pred, x, x_recon, lamda_recon)
        test_loss += loss.data[0] * x.size(0)  # sum up batch loss
        y_pred = y_pred.data.max(1)[1]
        y_true = target.data.max(1)[1]
        #print('y_pred y_true')
        #print(y_pred, y_true)
        correct += y_pred.eq(y_true).cpu().sum()
        if index % 10 == 0:
            print('Test Epoch: {}'.format(index * len(x)))
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_indices)
    return test_loss, test_acc

# plot_MNIST()
# decoder_net=Decoder(784)
#train_decoder_as_autoencoder(1, decoder_net)
# plot_squash()
#count_samples_per_class(train_loader, test_loader)


'''net = NeuralNetClassifier(
    Capsule_network,
    max_epochs=2,
    lr=0.001,
    #criterion=capsule_loss
)
X = mnist.data.reshape(-1,1,28,28)/255
y = mnist.target
y_tensor = torch.from_numpy(y)
y_one_hot = torch.zeros(y_tensor.size(0), 10).scatter_(1, y_tensor.type(torch.LongTensor).view(-1, 1), 1.).numpy()
print('x shape',X.shape)
#print('x',X[0])
#print('y',y)
print('x type',type(X))
print('y_type',type(y))
print('y shape',y.shape)
#print(y)
#net.fit(X.astype(float), y.astype(float))
#y_proba = net.predict_proba(X)
#print(y_proba)
from sklearn.model_selection import GridSearchCV


params = {
    'lr': [0.01, 0.02],
    'max_epochs': [10, 20],
}
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

gs.fit(X, y)
print(gs.best_score_, gs.best_params_)
'''


def get_file_name_reconstruction(learning_rate=1, num_routings=1, lambda_recon=0.0005, num_epochs = 5, test_accuracy = 0):

    # The dir-name for the TensorBoard log-dir.
    time_stamp = time.strftime("%d_%m_%Y-%H:%M:%S")
    s = time_stamp+"_acc_{0}_lr_{1:.0e}_routing-iterations_{2}_lambda_{3}_epochs_{4}"
    # Insert all the hyper-parameters in the dir-name.
    file_name = s.format(test_accuracy,
                        learning_rate,
                        num_routings,
                        lambda_recon,
                        num_epochs
                        )

    return file_name

dim_learning_rate = Real(low=1e-6, high=1e-4, prior='log-uniform',
                         name='learning_rate')

dim_lambda_recon = Real(low=4e-4, high=4e-1, prior='log-uniform',
                        name='lambda_recon')

dim_num_routings = Integer(low=2, high=3, name='num_routings')

dim_num_epochs = Integer(low=19, high=20, name='num_epochs')

dimensions = [dim_learning_rate, dim_lambda_recon, dim_num_routings, dim_num_epochs]

default_parameters = [1e-3, 0.0005, 3, 1]
best_accuracy = 0.0


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, lambda_recon, num_routings, num_epochs):
    print('Building network!')
    capsule_network = Capsule_network(num_routings)

    if use_cuda:
        print('Copying network to graphics card')
        capsule_network.cuda()
    print('Start!')
    print('learning rate: {0:.1e} lambda_recon: {1} num_routings: {2} num_epochs: {3}'.format(
                                                                                        learning_rate,
                                                                                        lambda_recon,
                                                                                        num_routings,
                                                                                        num_epochs))
    train(num_epochs, capsule_network, learning_rate, lamda_recon=LAMBDA_RECON, train_loader = train_loader)
    test_loss, test_acc = test(model=capsule_network, test_loader=test_loader, lamda_recon=LAMBDA_RECON)
    #test_acc = 100
    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    test_acc = float(test_acc)
    # If the classification accuracy of the saved model is improved ...
    global best_accuracy
    if test_acc > best_accuracy:
        # Save the new model to harddisk.
        # model.save(path_best_model)

        # Update the classification accuracy.
        file_name = get_file_name_reconstruction(learning_rate,num_routings, lambda_recon, num_epochs, test_acc)
        show_reconstruction(capsule_network, test_loader, 6, 'reconstructions', file_name)
        best_accuracy = test_acc
    print('best_accuracy', best_accuracy)
    print('test_accuracy', test_acc)
    print('test_loss', test_loss)
    del capsule_network
    return -test_acc

#show_batch(test_loader, 'batch','example_batch')
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',  # Expected Improvement.
                            n_calls=100)

# ,
# x0=default_parameters)
