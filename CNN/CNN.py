import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import Dataset


class Net(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.fc1 = nn.Linear(input_length, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(100)

    def forward(self, x, batchnorm=False):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        if batchnorm:
            x = self.norm(self.fc3(x))
        else:
            x = self.fc3(x)
        x = self.fc4(x)
        return x


def train_cnn(net, path, data_set, batch_size):
    torch.save(net.state_dict(), path + "NN_old.pth")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    train_loader = torch.utils.data.DataLoader(CustomTextDataset(np.array(data_set)), batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    last_loss = 0
    for epoch in range(25):  # loop over the dataset multiple times
        if epoch == 15:
            optimizer = optim.Adam(net.parameters(), lr=0.00001)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                if running_loss == last_loss:
                    break
                last_loss = running_loss
                running_loss = 0.0
        if running_loss == last_loss:
            break

    torch.save(net.state_dict(), path + "NN.pth")
    print('Finished Training')


def cnn_main(path, data_set, batch_size=10, actions=["train", "test"]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(actions) == 0:
        print("not actions given to cnn")
        return 0
    input_length = len(data_set[0]) - 1
    net = Net(input_length).to(device)
    # if os.path.isfile(path):
    #    net.load_state_dict(torch.load(path + "NN.pth"))

    train_set, test_set = torch.utils.data.random_split(data_set, [0.995, 0.005])
    if "train" in actions:
        train_cnn(net, path, train_set, batch_size)

    if "test" in actions:
        test_cnn(net, path, test_set)


def test_cnn(net, path, test_set):
    test_loader = torch.utils.data.DataLoader(CustomTextDataset(np.array(test_set)),
                                              shuffle=True, num_workers=2)
    total = 0
    correct = 0
    thresh = 0.1
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            predicted = outputs[0]
            total += 1
            if np.abs((labels - predicted) / (predicted)) < thresh:
                correct += 1
            elif total + 10 < len(test_set):
                print(str(labels) + "   " + str(predicted))
    print("percent correct: " + str(correct * 100 / total) + "%")


class CustomTextDataset(Dataset):

    def __init__(self, data):
        self.values = data[:, :-1]
        self.labels = data[:, -1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        values = self.values[idx]
        label = self.labels[idx]
        sample = values, label
        return sample
