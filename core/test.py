import torch


def test(extractor, classifier, data_loader):
    """ Estimate the model performance """

    # Load the models to GPU
    extractor = extractor.cuda()
    classifier = classifier.cuda()

    # Set the model to evaluation mode
    extractor.eval()
    classifier.eval()
    
    num_data = 0
    total_acc = 0.0

    # Test
    with torch.no_grad():
        for images, labels in data_loader:
            # Load images and labels to GPU
            images = images.cuda()
            labels = labels.cuda()

            # Predict the labels
            preds = classifier(extractor(images))

            # Update the total accuracy
            num_data += len(images)
            total_acc += (preds.max(1)[1] == labels).sum().item()

        total_acc = total_acc / num_data

    print('Test Accuracy: {:.4f}%'.format(total_acc * 100))
