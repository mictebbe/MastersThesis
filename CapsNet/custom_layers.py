# Imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# (Set Random seed)
torch.manual_seed(100)

s_j1 = torch.FloatTensor([1, 1, 1, 1])
s_j0 = torch.FloatTensor([0, 0, 0, 0])
s_j2 = torch.FloatTensor([5, 3, 6, 2])

def squash(s_j_=[], epsilon=1e-7, dim = -1):
    norm = torch.norm(s_j_)
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


def plot_squash():
    squashed_vector = torch.Tensor([])
    max = 100
    for i in range(0, 5):
        for j in range(0, max):
            v = torch.Tensor([j])/max + i  # t.rand(1) *40
            #print(v)
            v_squashed = squash(v)
            squashed_vector = torch.cat((squashed_vector, v_squashed), 0)
        #sorted_vector, indices = t.sort(squashed_vector)
    # print(sorted_vector)
    #test = t.Tensor([1,2,3,4,5,6],[1,2,3,4,5,6])
    plt.plot(squashed_vector.numpy())
    plt.show()

# (Conv Layers)
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

# Digit Capsules
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
        routing_weights = Variable(torch.zeros(x.size(0), self.num_caps_out, self.num_caps_in))#.cuda()
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