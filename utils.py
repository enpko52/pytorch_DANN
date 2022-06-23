import os
import yaml
import torch


def optimizer_scheduler(optimizer, p):
    """ Adjust the learning rate of optimizer """

    # Get the parameters for adjusting the learning rate
    config = yaml.load(open('config.yaml'))['lr']
    initial_lr = config['initial_lr']
    alpha = config['alpha']
    beta = config['beta']

    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr / (1. + alpha * p) ** beta

    return optimizer


def save_model(model, filename):
    """ Save the model parameters """

    # Get the directory name
    root = yaml.load(open('config.yaml'))['save']

    # Make the directory for saving model parameters
    if not os.path.exists(root):
        os.makedirs(root)

    # Save the model parameters
    torch.save(model.state_dict(), os.path.join(root, filename))


def load_model(model, filename):
    """ Load the model parameters """

    # Get the directory name
    root = yaml.load(open('config.yaml'))['save']
    filepath = os.path.join(root, filename)

    assert os.path.exists(filepath), 'There is no {}.'.format(filepath)

    # Load the model parameters
    model.load_state_dict(torch.load(filepath))

    return model
