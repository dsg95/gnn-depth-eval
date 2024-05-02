import torch
import torch.nn.functional as F
import numpy as np
from os import rename
from copy import deepcopy


def train_test_save(dataset, model, save_as='', partition=0,
                    epochs=200, patience=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = deepcopy(dataset).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), weight_decay=5e-4),
    ], lr=0.01)

    data.train_mask = data.train_mask[:, partition].to(torch.bool)
    data.val_mask = data.val_mask[:, partition].to(torch.bool)
    data.test_mask = data.test_mask[:, partition].to(torch.bool)

    model, best_test_acc = train(model, data, optimizer, epochs=epochs, patience=patience, save_as=save_as)

    if save_as != '':
        save_rename = save_as + '_TestAcc%0.2f' % best_test_acc
        rename(save_as, save_rename)
    print('Finished experiment %s' % save_as)

    return save_rename if save_as != '' else None


def train(model, data, optimizer, epochs=100, patience=50, save_as=''):

    early_stopping = EarlyStopping(patience=patience, min_delta=0.005)

    best_val_acc = 0.
    best_test = None
    best_model = None

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        out = model(data)
        train_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])  # nll_loss
        val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
        train_acc, val_acc, test_acc = test(model, data)
        print('Training\tEpoch %i\t\tTrain Loss: %.4f\t\tVal Loss: %.4f\t\tTrain Acc: %.4f'
              % (epoch, train_loss.item(), val_loss.item(), train_acc))

        if val_acc > best_val_acc:
            print('New best model | Test Accuracy %0.2f' % test_acc)
            best_test = test_acc
            best_val_acc = val_acc
            best_model = model
            try:
                model.eval()
                _ = model(data, save_as=save_as)
            except TypeError:
                print('Cannot save test data. Proceeding without saving.')
                _ = model(data).argmax(dim=1)

        # early stopping
        if early_stopping.early_stop(val_loss):
            print("Ending with early stopping at epoch", epoch)
            print('Best model test:', best_test)
            break

        train_loss.backward()
        optimizer.step()
    return best_model, best_test


def get_acc(pred, y):
    correct = (pred == y).sum()
    acc = int(correct) / len(y)
    return acc


@torch.no_grad()
def test(model, data, save_as=''):
    model.eval()
    try:
        pred = model(data, save_as=save_as).argmax(dim=1)
    except TypeError:
        print('Cannot save test data. Proceeding without saving.')
        pred = model(data).argmax(dim=1)
    train_acc = get_acc(pred[data.train_mask], data.y[data.train_mask])
    val_acc = get_acc(pred[data.val_mask], data.y[data.val_mask])
    test_acc = get_acc(pred[data.test_mask], data.y[data.test_mask])

    print('Val accuracy: %.4f\t\tTest accuracy: %.4f' % (val_acc, test_acc))

    if save_as != '':
        rename(save_as, save_as + '_TestAcc%0.2f' % test_acc)

    return train_acc, val_acc, test_acc


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
