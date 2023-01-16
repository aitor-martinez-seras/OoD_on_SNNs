from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np

from test import validate_one_epoch


def train_one_epoch(model, device, train_loader, optimizer):

    model.train()
    losses = []

    for (data, target) in tqdm(train_loader, leave=False, desc='Progress of one epoch'):

        # Process data
        data, target = data.to(device), target.to(device)
        output = model(data)

        # Negative loglikelihoog loss, for classification problems.
        # The input must contain log probabilities of each class
        loss = torch.nn.functional.nll_loss(output, target)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Store the losses of every minibatch
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


def train(model, device, train_loader: DataLoader, test_loader: DataLoader, optimizer: Optimizer, epochs=10):
    training_losses = []
    test_losses = []
    accuracies = []
    for epoch in trange(epochs, desc='Number of epochs', leave=False):
        _, mean_training_loss = train_one_epoch(model, device, train_loader, optimizer)
        mean_test_loss, accuracy, _ = validate_one_epoch(model, device, test_loader)
        training_losses.append(mean_training_loss)
        test_losses.append(mean_test_loss)
        accuracies.append(accuracy)
        print(f"\nThe accuracy of the model with {train_loader.dataset.__str__()} "
              f"for epoch {epoch + 1} is {accuracies[-1]}%")
