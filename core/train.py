import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import optimizer_scheduler, save_model


def source_only(extractor, classifier, source_loader):
    """ Train the models using only source dataset """

    # Get the parameters for training
    config = yaml.load(open('config.yaml'))

    # Load the models to GPU
    extractor = extractor.cuda()
    classifier = classifier.cuda()

    # Set up criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(params=list(extractor.parameters()) + list(classifier.parameters()),
                          lr=config['lr']['initial_lr'],
                          momentum=config['momentum'])

    # Training
    print('\nSource-Only Training...\n')

    for epoch in range(config['epochs']):
        # Set the model to train mode
        extractor.train()
        classifier.train()

        num_data = 0
        total_acc = 0.0
        total_loss = 0.0

        for idx, (images, labels) in enumerate(source_loader):
            # Update the learning rate
            p = (idx + epoch * len(source_loader)) / config['epochs'] / len(source_loader)
            optimizer = optimizer_scheduler(optimizer, p)
            
            # Load images and labels to GPU
            images = images.cuda()
            labels = labels.cuda()

            # Predict labels and compute loss
            preds = classifier(extractor(images))
            loss = criterion(preds, labels)

            # Optimize the models
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total loss and total accuracy
            num_data += len(images)
            total_acc += (preds.max(1)[1] == labels).sum().item()
            total_loss += loss.item()

        total_acc = total_acc / num_data
        total_loss = total_loss / len(source_loader)

        # Print log information
        print('Epoch [{:4}/{:4}] Loss: {:8.4f}, Accuracy: {:.4f}%'.format(
            epoch+1, config['epochs'], total_loss, total_acc * 100
        ))

        # Save the model parameters
        if (epoch + 1) % 10 == 0:
            save_model(extractor, 'source_extractor_{}.pt'.format(epoch+1))
            save_model(classifier, 'source_classifier_{}.pt'.format(epoch+1))

    return extractor, classifier


def dann(extractor, classifier, discriminator, source_loader, target_loader):
    """ Train the models of DANN """

    # Get the parameters for training
    config = yaml.load(open('config.yaml'))

    # Load the models to GPU
    extractor = extractor.cuda()
    classifier = classifier.cuda()
    discriminator = discriminator.cuda()

    # Set up criterion and optimizer
    cls_criterion = nn.CrossEntropyLoss().cuda()
    dis_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(params=list(extractor.parameters()) + 
                                 list(classifier.parameters()) +
                                 list(discriminator.parameters()),
                          lr=config['lr']['initial_lr'],
                          momentum=config['momentum'])

    # Training
    print('\nDANN Training...\n')

    for epoch in range(config['epochs']):
        # Set the model to train mode
        extractor.train()
        classifier.train()
        discriminator.train()

        num_data = 0
        total_acc = 0.0
        total_loss = 0.0
        total_cls_loss = 0.0
        total_dis_loss = 0.0
        len_loader = min(len(source_loader), len(target_loader))

        for idx, (src_data, tgt_data) in enumerate(zip(source_loader, target_loader)):
            src_images, src_labels = src_data
            tgt_images, _ = tgt_data

            # Compute the alpha value and update the learning rate
            p = (idx + epoch * len_loader) / config['epochs'] / len_loader
            alpha = 2. / (1. + np.exp(-config['gamma'] * p)) - 1
            optimizer = optimizer_scheduler(optimizer, p)

            # Load images and labels to GPU
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            tgt_images = tgt_images.cuda()

            # Predict class labels and compute classification loss
            cls_preds = classifier(extractor(src_images))
            cls_loss = cls_criterion(cls_preds, src_labels)

            # Update total classification loss and total classification accuracy
            num_data += len(src_images)
            total_acc += (cls_preds.max(1)[1] == src_labels).sum().item()
            total_cls_loss += cls_loss.item()


            # Make the domain labels
            domain_source_labels = torch.zeros(src_images.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(tgt_images.shape[0]).type(torch.LongTensor)
            domain_labels = torch.cat([domain_source_labels, domain_target_labels], 0).cuda()
            combined_images = torch.cat([src_images, tgt_images], 0)

            # Predict domain labels and compute discrimination loss
            dis_preds = discriminator(extractor(combined_images), alpha)
            dis_loss = dis_criterion(dis_preds, domain_labels)

            # Update total discrimination loss and total loss
            loss = cls_loss + dis_loss
            total_dis_loss += dis_loss.item()
            total_loss += loss.item()

            # Optimize the models
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
  
        total_acc = total_acc / num_data
        total_loss = total_loss / len_loader
        total_cls_loss = total_cls_loss / len_loader
        total_dis_loss = total_dis_loss / len_loader

        # Print log information
        print('Epoch [{:4}/{:4}] Loss: {:8.4f}, Class Loss: {:8.4f}, Domain Loss: {:8.4f}, Accuracy: {:.4f}%'.format(
            epoch+1, config['epochs'], total_loss, total_cls_loss, total_dis_loss, total_acc * 100
        ))

        # Save the model parameters
        if (epoch + 1) % 10 == 0:
            save_model(extractor, 'dann_extractor_{}.pt'.format(epoch+1))
            save_model(classifier, 'dann_classifier_{}.pt'.format(epoch+1))

    return extractor, classifier
