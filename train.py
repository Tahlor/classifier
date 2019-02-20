import numpy as np
import torch
import pandas as pd
from timeit import default_timer as timer
import torch.nn as nn
from torchvision import models
from PIL import Image
import os
import cutils

def initialize_model(model_name="vgg16", n_classes=None, train_on_gpu=True, multi_gpu=False):

    final_layers = lambda n_inputs : nn.Sequential(
                      nn.Linear(n_inputs, 2048),
                      nn.ReLU(),
                      nn.Dropout(0.2), # probability of being 0'd
                      nn.Linear(2048, 2048),
                      nn.ReLU(),
                      nn.Dropout(0.2),  # probability of being 0'd
                      nn.Linear(2048, n_classes))

    if "resnet101_full" == model_name:
        model = models.resnet101(pretrained=True)
        n_inputs = model.fc.in_features
        model.fc = final_layers(n_inputs)

    elif "resnet" in model_name:
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
        elif model_name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=True)
        n_inputs = model.fc.in_features

        # Turn off backprop
        for param in model.parameters():
            param.requires_grad = False
        model.fc = final_layers(n_inputs)

    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        n_inputs = model.classifier[6].in_features

        # Turn off backprop
        for param in model.parameters():
            param.requires_grad = False

        model.classifier[6] = final_layers(n_inputs)

    elif model_name == "vgg16_full":
        model = models.vgg16(pretrained=True)
        n_inputs = model.classifier[6].in_features
        model.classifier[6] = final_layers(n_inputs)

    elif model_name == "squeezenet":
        model = models.squeezenet1_1(pretrained=True)
        model_conv = models.squeezenet1_1()
        for name, params in model_conv.named_children():
            print(name)

        ## How many In_channels are there for the conv layer
        in_ftrs = model_conv.classifier[1].in_channels
        ## How many Out_channels are there for the conv layer
        out_ftrs = model_conv.classifier[1].out_channels
        ## Converting a sequential layer to list of layers
        features = list(model_conv.classifier.children())
        ## Changing the conv layer to required dimension
        features[1] = nn.Conv2d(in_ftrs, n_classes, kernel_size, stride) # is n_class right here?
        ## Changing the pooling layer as per the architecture output
        features[3] = nn.AvgPool2d(12, stride=1)
        ## Making a container to list all the layers
        model_conv.classifier = nn.Sequential(*features)
        ## Mentioning the number of out_put classes
        model_conv.num_classes = n_classes

    # Loop through model
    # for child in model.children():


    # output layer


    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    model.model_name = model_name
    return model

def calc_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res[0]

def validate(model, criterion, valid_loader, train_on_gpu=True):
    # Don't need to keep track of gradients
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()
        valid_loss = 0
        valid_acc = 0
        valid_acc5 = 0
        items = 0
        for data, target in valid_loader: # load 1 batch
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Forward pass
            output = model(data)

            # Validation loss
            loss = criterion(output, target)
            # Multiply average loss times the number of examples in batch
            valid_loss += loss.item() * data.size(0)

            # Calculate validation accuracy
            # _, pred = torch.max(output, dim=1)
            # correct_tensor = pred.eq(target.data.view_as(pred))
            # acc1 = torch.mean(correct_tensor.type(torch.FloatTensor))
            acc = calc_accuracy(output, target, topk=(1,))
            acc5 = calc_accuracy(output, target, topk=(5,))
            # Multiply average accuracy times the number of examples
            valid_acc += acc.item() * data.size(0)
            valid_acc5 += acc5.item() * data.size(0)
            items += data.size(0)
            #valid_acc5 += acc5.item() * data.size(0)
        print(f'Loss: {valid_loss/items:.4f}')
        print(f'Accuracy: {valid_acc/items:.4f}')
        print(f'Accuracy5: {valid_acc5/items:.4f}')
    return valid_acc, valid_acc5, valid_loss

def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=7,
          n_epochs=20,
          print_every=2,
          train_on_gpu=True,
          early_stopping=True,
          scheduler=None,
          interval=5000):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []
    model.optimizer = optimizer

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')
        save_checkpoint(model, save_file_name)

    overall_start = timer()

    # Main loop
    for epoch in range(model.epochs, n_epochs+model.epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        valid_loader = train_loader if valid_loader is None else valid_loader

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            if ii % interval == 0:
                #print("Training loss: {}".format(train_loss/ii))
                pass
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # Advance epoch
        if model.scheduler:
            model.scheduler.step()
        model.epochs += 1

        # Validate loop
        if valid_loader is None or train_loader == valid_loader:
            valid_acc = train_acc
            valid_loss = train_loss
        elif epoch % 5 == 0:
            print("Running validation set")
            valid_acc, valid_acc5, valid_loss = validate(model=model, criterion=criterion, valid_loader=valid_loader, train_on_gpu=train_on_gpu)

        # Calculate average losses
        valid_loss = valid_loss / len(valid_loader.dataset)
        train_loss = train_loss / len(train_loader.dataset)

        # Calculate average accuracy
        train_acc = train_acc / len(train_loader.dataset)
        valid_acc = valid_acc / len(valid_loader.dataset)

        history.append([train_loss, valid_loss, train_acc, valid_acc])

        # Print training and validation results
        if (epoch + 1) % print_every == 0:
            print(
                f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
            )
            print(
                f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
            )

        save_checkpoint(model, save_file_name)
        # Save the model if validation loss decreases
        if valid_loss < valid_loss_min:

            # Save model
            torch.save(model.state_dict(), save_file_name+"_partial")
            # Track improvement
            epochs_no_improve = 0
            valid_loss_min = valid_loss
            valid_best_acc = valid_acc
            best_epoch = epoch

        # Otherwise increment count of epochs with no improvement
        else:
            epochs_no_improve += 1
            # Trigger early stopping
            if epochs_no_improve >= max_epochs_stop and early_stopping:
                print(
                    f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                )
                total_time = timer() - overall_start
                print(
                    f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                )

                # Load the best state dict
                model.load_state_dict(torch.load(save_file_name))
                # Attach the optimizer
                model.optimizer = optimizer

                # Format history
                history = pd.DataFrame(
                    history,
                    columns=[
                        'train_loss', 'valid_loss', 'train_acc',
                        'valid_acc'
                    ])
                return model, history

        # try:
        #     h =  pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
        #     plot_loss(h)
        # except(Exception) as e:
        #     print(e)
        #     print("Your plot function sucks!")

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

    return model, history


def plot_loss(history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'train_acc']:
        plt.plot(
            history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.savefig("./loss.png")

def save_checkpoint(model, path):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`
    """

    ## This is a hack, should be if multi GPU
    gpu = next(model.parameters()).is_cuda
    if gpu:
        try:
            state_dict = model.module.state_dict()
        except:
            state_dict = model.state_dict()
    print("Saving model {}".format(model.model_name))

    model_parallel = model.module if gpu else model

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
        'model_name': model.model_name,
        'scheduler':model.scheduler
    }

    ## Add model to path
    if model.model_name not in path:
        print("Adding model name to path")
        path += model.model_name

    ## Extract the final classifier and the state dictionary
    if "_full" in model.model_name:
        checkpoint['model'] = model_parallel
    elif model.model_name == 'vgg16':
        # Check to see if model was parallelized
            checkpoint['classifier'] = model_parallel.classifier

    elif "resnet" in model.model_name:
        checkpoint['fc'] = model_parallel.fc
    else:
        print("Unknown model, saving the whole thing")
        checkpoint['model'] = model

    # Add the optimizer
    checkpoint['state_dict'] = state_dict
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    if os.path.isdir(path):
        cutils.mkdir(path)
        path = cutils.increment_path(model.model_name, path)
    torch.save(checkpoint, path)

def load_checkpoint(path, train_on_gpu=True, multi_gpu=False):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Path or directory of checkpoints

    Returns
    --------
        None, save the `model` to `path`

    """

    # Check if directory, if yes, find the one with the biggest number in it
    if not os.path.exists(path):
        cutils.mkdir(path)
    if os.path.isdir(path):
        _, path = cutils.get_max_file(path, ignore="partial")
        if os.path.isdir(path):
            print("No checkpoint found")
            return None, None

    # Load in checkpoint
    checkpoint = torch.load(path)

    if "vgg16_full" in path:
        model_name = "vgg16_full"
    elif "resnet101_full" in path:
        model_name = "resnet101_full"
    elif "vgg16" in path:
        model_name = "vgg16"
    elif "resnet18" in path:
        model_name = "resnet18"
    elif "resnet50" in path:
        model_name = "resnet50"
    elif "resnet101" in path:
        model_name = "resnet101"
    else:
        raise Exception("Unknown pretrained model, should be in checkpoint path")

    if "_full" in model_name:
        model = checkpoint['model']
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)

        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif "resnet" in model_name:
        if model_name=="resnet18":
            model = models.resnet18(pretrained=True)
        if model_name=="resnet50":
            model = models.resnet50(pretrained=True)
        if model_name == "resnet101":
            model = models.resnet101(pretrained=True)

        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']
    else:
        model = checkpoint['model']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    if multi_gpu:
        model = nn.DataParallel(model)

    if train_on_gpu:
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.scheduler = checkpoint['scheduler']
    model.epochs = checkpoint['epochs']
    model.model_name = checkpoint['model_name']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def predict(image_path, model, topk=5, train_on_gpu=True):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns

    """
    real_class = image_path.split('/')[-2]

    # Convert to pytorch tensor
    img_tensor = process_image(image_path)

    # Resize
    if train_on_gpu:
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class

def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor