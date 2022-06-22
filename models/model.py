import torch.nn as nn

from .functions import ReverseLayerF


class Extractor(nn.Module):
    """ The neural network class for extracting feature maps  """

    def __init__(self, in_channels):

        super(Extractor, self).__init__()
        self.in_channels = in_channels

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """ The method for forward propagation """

        x = self.extractor(x)
        x = x.view(x.shape[0], -1)
        return x


class Classifier(nn.Module):
    """ The neural network class for classifying labels """

    def __init__(self, in_features, out_features):

        super(Classifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=100),
            nn.ReLU(),

            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),

            nn.Linear(in_features=100, out_features=self.out_features)
        )

    def forward(self, x):
        """ The method for forward propagation """

        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    """ The neural network class for discriminating domain label """

    def __init__(self, in_features):

        super(Discriminator, self).__init__()
        self.in_features = in_features
        self.out_features = 2

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=100),
            nn.ReLU(),

            nn.Linear(in_features=100, out_features=self.out_features)
        )

    def forward(self, x, alpha):
        """ The method for forward propagation """

        x = ReverseLayerF.apply(x, alpha)
        x = self.discriminator(x)
        return x
