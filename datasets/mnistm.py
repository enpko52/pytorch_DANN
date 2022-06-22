import os
import gzip
import pickle
import requests

import yaml
import torch
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms


class MNISTM(data.Dataset):
    """ The MNIST-M dataset class """

    url = 'https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz'

    def __init__(self, root, train=True, download=False, transform=None):

        super(MNISTM, self).__init__()
        self.root = root
        self.data_dir = 'MNISTM'
        self.raw_dir = 'raw'
        self.train = train
        self.transform = transform

        if download:
            self.download()

    def __getitem__(self, index):
        """ Get images and target for data loader """

        if self.train:
            image, target = self.train_images[index], self.train_labels[index]
        else:
            image, target = self.test_images[index], self.test_labels[index]

        image = Image.fromarray(image.squeeze().numpy(), mode='RGB')

        # Pre-processing
        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        """ Return size of dataset """

        if self.train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def download(self):
        """ Download the MNIST-M data """

        # Make data directory
        os.makedirs(os.path.join(self.root, self.data_dir, self.raw_dir), exist_ok=True)

        # Download the pkl file
        filename = self.url.split('/')[-1]
        filepath = os.path.join(self.root, self.data_dir, self.raw_dir, filename)

        if not os.path.exists(filepath):
            print('Downloading {}'.format(self.url))
            
            response = requests.get(self.url)
            open(filepath, 'wb').write(response.content)

            # Extract pkl file from gz file
            with open(filepath.replace('.gz', ''), 'wb') as f:
                f.write(gzip.open(filepath, 'rb').read())
        
        # Load MNIST-M images from pkl file
        with open(filepath.replace('.gz', ''), 'rb') as f:
            mnistm_data = pickle.load(f, encoding='bytes')

        self.train_images = torch.ByteTensor(mnistm_data[b'train'])
        self.test_images = torch.ByteTensor(mnistm_data[b'test'])

        # Get MNIST-M labels from MNIST dataset
        self.train_labels = datasets.MNIST(root=self.root,
                                           train=True,
                                           download=True).targets
        self.test_labels = datasets.MNIST(root=self.root,
                                          train=False,
                                          download=True).targets


def get_mnistm(train=True):
    """ Get the MNIST-M data loader """

    # Get the parameters for creating data loader
    config = yaml.load(open('config.yaml'))

    # Image pre-processing
    transform = transforms.Compose([transforms.Resize(config['img_size']),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # MNIST-M dataset
    mnistm = MNISTM(root=config['root'],
                    train=train,
                    download=True,
                    transform=transform)

    # MNIST-M data loader
    mnistm_loader = data.DataLoader(dataset=mnistm,
                                    batch_size=config['batch_size'],
                                    shuffle=True)
    
    return mnistm_loader
