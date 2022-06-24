import os
import yaml
import argparse
import warnings

from utils import load_model, visualize
from core import source_only, dann, test
from datasets import get_mnist, get_mnistm
from models import Extractor, Classifier, Discriminator


MODE_MAP = {'source-only': 'Source-Only', 'dann': 'DANN'}
DATASETS_MAP = {'mnist': 'get_mnist', 'mnistm': 'get_mnistm'}


# Ignore warnings
warnings.filterwarnings(action='ignore')


def get_args():
    """ Get the arguments for training and test """

    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='mnist', choices=DATASETS_MAP.keys(), help='Source datasets')
    parser.add_argument('--target', type=str, default='mnistm', choices=DATASETS_MAP.keys(), help='Target datasets')
    parser.add_argument('--mode', type=str, default='dann', choices=MODE_MAP.keys(), help='Training mode')
    parser.add_argument('--train', action='store_true', help='Train the models')
    parser.add_argument('--extractor', type=str, default=None, help='Extractor\'s weights file')
    parser.add_argument('--classifier', type=str, default=None, help='Classifier\'s weights file')

    args = parser.parse_args()
    return args


def main(args):
    """ The main function """

    # Get the parameters
    config = yaml.load(open('config.yaml'))

    # Get the datasets
    train_source_loader = eval(DATASETS_MAP[args.source])(train=True)
    train_target_loader = eval(DATASETS_MAP[args.target])(train=True)
    test_source_loader = eval(DATASETS_MAP[args.source])(train=False)
    test_target_loader = eval(DATASETS_MAP[args.target])(train=False)

    # Get the models
    extractor = Extractor(**config['extractor'])
    classifier = Classifier(**config['classifier'])
    discriminator = Discriminator(**config['discriminator'])
    
    # Training
    if args.train:
        if args.mode == 'source-only':
            extractor, classifier = source_only(extractor, classifier, train_source_loader)
        else:
            extractor, classifier = dann(extractor, classifier, discriminator, train_source_loader, train_target_loader)

    # Load the models
    else:
        assert args.extractor != None, 'If train is False, you have to input the weights file.'
        assert args.classifier != None, 'If train is False, you have to input the weights file.'

        ext_filepath = os.path.join(config['save'], args.extractor)
        cls_filepath = os.path.join(config['save'], args.classifier)

        assert os.path.exists(ext_filepath), 'There is no {}'.format(ext_filepath)
        assert os.path.exists(cls_filepath), 'There is no {}'.format(cls_filepath)

        extractor = load_model(extractor, args.extractor)
        classifier = load_model(classifier, args.classifier)

    # Test
    print('\nTest Result with Source Datasets on {}\n'.format(MODE_MAP[args.mode]))
    test(extractor, classifier, test_source_loader)

    print('\nTest Result with Target Datasets on {}\n'.format(MODE_MAP[args.mode]))
    test(extractor, classifier, test_target_loader)

    # Visualization
    print('\nVisualizing...\n')
    visualize(extractor, test_source_loader, test_target_loader, MODE_MAP[args.mode] + '.png')

    print('Done!')
    

if __name__ == '__main__':
    args = get_args()
    main(args)