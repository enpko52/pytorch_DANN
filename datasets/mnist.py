import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist(train=True):
    """ Get the MNIST data loader """

    # Get the parameters for creating data loader
    config = yaml.load(open('config.yaml'))

    # Image pre-processing
    transform = transforms.Compose([transforms.Resize(config['img_size']),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

    # MNIST dataset
    mnist = datasets.MNIST(root=config['root'],
                           train=train,
                           download=True,
                           transform=transform)

    # MNIST data loader
    mnist_loader = DataLoader(dataset=mnist,
                              batch_size=config['batch_size'],
                              shuffle=True)
    
    return mnist_loader
