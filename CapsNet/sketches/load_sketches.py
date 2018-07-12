from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import numbers
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.ndimage.morphology import binary_dilation, grey_dilation, grey_erosion
import torch.nn.functional as F
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

SKETCH_PATH = 'sketch_dataset/png/'
#SKETCH_PATH = '/Users/michaeltebbe/Masterarbeit/Masterarbeit/CapsNet/sketches/sketch_dataset/png/'
SKETCH_LIST = 'filelist.txt'
torch.manual_seed(42)
class SketchDataset(Dataset):
    """TU Berlin Sketch dataset."""

    def __init__(self, file_list, root_dir, transform=None, categories = None):
        """
        Args:
            file_list (string): Name of the Filelist.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            categories ([], optional): get only the categories in the list as dataset; list is ordered alphabetically by SketchDataset for consistency
        """
        #load filelist.txt as dataframe
        self.sketches_frame = pd.read_csv(root_dir + file_list)
        #set members
        self.root_dir = root_dir
        self.transform = transform
        #create set of categories
        self.categories = set()
        labels = []
        for idx, _ in self.sketches_frame.iterrows():
            path_string = self.sketches_frame.iloc[idx, 0]
            index = path_string.find('/')
            label = path_string[:index]
            self.categories.add(label)
            labels.append(label)
        
        #cast self.categories to list to get indices
        self.categories = list(self.categories)
        #add 'labels' column to dataframe
        self.labels_frame = pd.DataFrame({'label': labels})
        
        self.sketches_frame = pd.concat([self.sketches_frame, self.labels_frame], axis=1)

        

        #filter dataset for categories given as parameter
        if categories is not None:
            self.categories = categories
            self.sketches_frame = self.sketches_frame.loc[self.sketches_frame['label'].isin(self.categories)]
        #order self.categories alphabetically for consistency
        self.categories = sorted(self.categories)
        
        #create label_index column, mapping labels to index in self.categories
        self.categories = dict( zip( self.categories, range(0, len(self.categories))))
        self.sketches_frame['label_index'] = self.sketches_frame['label'].map(self.categories)



        

    def __len__(self):
        return len(self.sketches_frame)

    def __getitem__(self, idx):
        path_string = self.sketches_frame.iloc[idx, 0]
        img_name = os.path.join(self.root_dir,
                                path_string)
        image = Image.open(img_name)
        index = path_string.find('/')
        label = self.sketches_frame.iloc[idx, 1]
        label_index = self.sketches_frame.iloc[idx, 2]
        sample = {'image': image, 'label': label, 'label_index':label_index}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            
        return sample

class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, [self.padding,self.padding])

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        mode = img.mode
        img = img.convert('RGBA')
        # rotated image
        rotated_img = F.crop(img, i, j, h, w)
        # a white image same size as rotated image
        black_img = Image.new('RGBA', rotated_img.size, (255,255,255,255))
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rotated_img, black_img, rotated_img)
        # save your work (converting back to mode='1' or whatever..)
        out = out.convert(mode)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def show_sketches(sample):
    ax = plt.subplot(1, 1, 1)
    
    ax.set_title('{} {}'.format(sample['label'][0], sample['label_index'][0]))
    ax.axis('off')
    plt.imshow(sample['image'][0])
    plt.show()

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img):

        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(img, (new_h, new_w))
        #print('img rescale', image.shape)
        return img

#From Torchvision 0.4.0
class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)
        #width, height = img.size
        #new_img = Image.new('L', (width, height), (0))
        #img = img.rotate(angle, self.resample, self.expand)
        #new_img = new_img.paste(img)

        # original image
        # converted to have an alpha layer
        mode = img.mode
        img = img.convert('RGBA')
        # rotated image
        rotated_img = img.rotate(22.2, expand=1)
        # a white image same size as rotated image
        black_img = Image.new('RGBA', rotated_img.size, (255,255,255,255))
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rotated_img, black_img, rotated_img)
        # save your work (converting back to mode='1' or whatever..)
        out = out.convert(mode)
        return out

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        format_string += ')'
        return format_string


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        return torch.from_numpy(img.astype(np.float64))

class ToNumpy(object):
    """Convert ndarrays in sample to np.ndarray."""
    def __call__(self, img):
        return np.array(img)
               

class Dilate(object):
    """Dilate Images to preserve lines after rescaling."""

    def __call__(self, img):
        return binary_dilation(img)

class Invert(object):
    """Convert ndarrays in sample to Tensors."""

    

    def __call__(self, img):
        def invert(img, threshold):
            return (threshold >= img)
        img = invert(img, 0.5)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        return img
    
    


class ToInt(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        return (img == 1).astype(np.int64)
'''
mean = 0
std = 0
transforms_composed = transforms.Compose([
    Invert(),
    Dilate(),
    Dilate(),
    Dilate(),
    Dilate(),
    Rescale(128),
    ToInt(),
    ToTensor()])#, Normalize()])
sketch_dataset = SketchDataset(file_list = SKETCH_LIST, root_dir = SKETCH_PATH,categories = ['cat', 'dog', 'butterfly','wheel', 'wine-bottle','apple', 'book', 'moon', 'pig', 'ship'], transform = transforms_composed)
#sketch_dataset = SketchDataset(file_list = SKETCH_LIST, root_dir = SKETCH_PATH, transform = transforms_composed)

#fig = plt.figure()

for i in range(len(sketch_dataset)):
    sample = sketch_dataset[i]
    x = 5
    y = 1

    print(i, sample['image'].shape, sample['label'], sample['label_index'])

    ax = plt.subplot(y, x, i + 1)
    plt.tight_layout()
    ax.set_title('#{}: {} {}'.format(i, sample['label'], sample['label_index']))
    ax.axis('off')
    show_sketches(**sample)

    if i == (x*y)-1:
        #plt.show()
        break
'''
'''
batch_size = 16
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

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(sketch_dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(sketch_dataset, batch_size=batch_size,
sampler=test_sampler)
'''

'''
once =True
#Count categories
category_count=np.zeros(len(sketch_dataset.categories))

for batch_index, (sample) in enumerate(train_loader):
    #Count categories
    for i in sample['label_index']:
        #print(i)
        category_count[i]+=1
    

    if once:
        once = False
        ax = plt.subplot(1, 1, 1)
        ax.set_title('{} {}'.format(sample['label'][0], sample['label_index'][0]))
        ax.axis('off')
        sumnot0 = (sample['image'][0] != 0).sum()
        print('sumnot0', sumnot0)
        sumnot1 = (sample['image'][0] != 1).sum()
        print('sumnot1', sumnot1)

        sum0 = (sample['image'][0] == 0).sum()
        print('sum0', sum0)
        sum1 = (sample['image'][0] == 1).sum()
        print('sum1', sum1)

        print('sum all',sample['image'][0].sum())
        plt.imshow(sample['image'][0])
    
#print(sketch_dataset.categories)
#Count categories print (category_count)


plt.show()
'''