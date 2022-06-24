import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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


def _plot_graph(features, labels, domain, filename):
    """ Plot the t-SNE graph """

    # Make the visualization directory
    root = yaml.load(open('config.yaml'))['visual_root']
    
    if not os.path.exists(root):
        os.mkdir(root)

    # Rescale the feature range
    feat_max, feat_min = np.max(features, 0), np.min(features, 0)
    features = (features - feat_min) / (feat_max - feat_min)

    # Plotting
    color = {0: 'r', 1: 'b'}

    plt.figure(figsize=(10, 10))
    plt.title(filename.split('.')[0], fontsize=20)

    for i in range(features.shape[0]):
        plt.text(features[i][0], features[i][1], 
                 str(labels[i]), 
                 color=color[domain[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(root, filename))


def visualize(extractor, source_loader, target_loader, filename):
    """ Visualize the data distribution using t-SNE """

    images = []
    labels = []
    domain = []

    # Get some samples from the data loader
    for idx, (src_data, tgt_data) in enumerate(zip(source_loader, target_loader)):
        if idx >= 15:
            break
        
        images.extend(src_data[0].tolist())
        images.extend(tgt_data[0].tolist())

        labels.extend(src_data[1].tolist())
        labels.extend(tgt_data[1].tolist())

        domain.extend([0] * src_data[0].shape[0])
        domain.extend([1] * tgt_data[0].shape[0])

    # Load a model and images to GPU
    extractor = extractor.cuda()
    images = torch.tensor(images).cuda()

    # Extract the feature maps
    features = extractor(images)

    # Reduce the feature dimensions
    tsne = TSNE(n_components=2, perplexity=30., n_iter=3000, init='pca')
    features = tsne.fit_transform(features.detach().cpu().numpy())

    # Plotting
    _plot_graph(features, labels, domain, filename)
