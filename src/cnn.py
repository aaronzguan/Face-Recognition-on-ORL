import argparse
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR



parser = argparse.ArgumentParser(description='ORL Neural Network')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=5, type=int, help='')
parser.add_argument('--num_epochs', default=100, type=int, help='')
parser.add_argument('--model_root', default='/home/aaron/ORLFace/data/ORLnet_18_11_29.pkl')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on', device)


class ORLNet(nn.Module):
    """
    Create convolution neural network
    Conv.1: [5x5, 16], S1
    Pool.1: [2x2] S2, maximum pooling
    Conv.2: [5x5, 32], S1
    Pool.2: [2x2] S2, maximum pooling
    FC1: 32 * 25 * 20
    FC2: 1024
    FC3: 1024
    """
    def __init__(self, channel, num_classes=40):
        super(ORLNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 25 * 20, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 25 * 20)
        x = self.classifier(x)
        return x


class ORLDataset(torch.utils.data.Dataset):
    def __init__(self, train_x, train_y, text_x, text_y, is_train=True):
        super(ORLDataset, self).__init__()
        if is_train:
            self.data = train_x
            self.label = train_y
        else:
            self.data = text_x
            self.label = text_y

    def __getitem__(self, item):
        img = self.data[item].reshape(112, 92, 1)
        target = self.label[item]
        return ToTensor()(img), torch.LongTensor(target)

    def __len__(self):
        return len(self.data)


def train(net, loader, optimizer, criterion, epoch):

    net.train()

    print('Epoch {}/{}'.format(epoch+1, args.num_epochs))
    print('-' * 10)

    running_batch = 0
    running_loss = 0.0
    running_corrects = 0

    # Iterate over images.
    for i, (face, label) in enumerate(loader):

        face = face.to(device)
        label = label.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        output = net(face)
        _, label_pred = torch.max(output, 1)

        loss = criterion(output, label.view(-1))
        loss.backward()
        optimizer.step()

        running_batch += label.size(0)
        running_loss += loss.item() * face.size(0)
        running_corrects += torch.sum(label_pred == label.view(-1)).item()

        if (i+1) % 5 == 0:  # print every 5 mini-batches
            print('[%d, %5d] loss: %.3f correct: %.3f' % (epoch + 1, i + 1, running_loss/running_batch, running_corrects/running_batch))


def test(net, loader, criterion,epoch):

    net.eval()

    running_batch = 0
    running_loss = 0.0

    label_truth = []
    label_output = []

    with torch.no_grad():
        for i, (face, label) in enumerate(loader):
            face = face.to(device)
            label = label.to(device)
            output = net(face)

            label_pred = torch.nn.functional.softmax(output, dim=1)
            loss = criterion(output, label.view(-1))
            running_batch += label.size(0)
            running_loss += loss.item() * face.size(0)
            label_truth.extend(label.view(-1).cpu().numpy())
            label_output.extend(label_pred.cpu().numpy())

    label_output = np.argmax(label_output, axis=1)
    accuracy = accuracy_score(label_truth, label_output)
    print('*' * 40)
    print('Epoch: %d, average_precision_val: %.3f' % (epoch, accuracy))
    print('*' * 40)
    # print(classification_report(label_truth, label_output))


def cnn_face_recognition(train_X, train_Y, test_X, test_Y):

    torch.set_default_tensor_type('torch.DoubleTensor')
    train_data = ORLDataset(train_X, train_Y, test_X, test_Y, is_train=True)
    train_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=args.batch_size, pin_memory=False,
                              shuffle=True)
    test_data = ORLDataset(train_X, train_Y, test_X, test_Y, is_train=False)
    test_loader = DataLoader(dataset=test_data, num_workers=4, batch_size=args.batch_size, pin_memory=False,
                             shuffle=False)

    net = ORLNet(channel=1, num_classes=40)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.005)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    # ------------------------
    # Start Training and Validating
    # ------------------------
    save_step = 100
    for epoch in range(args.num_epochs):
        scheduler.step()
        train(net, train_loader, optimizer, criterion, epoch)
        print('Testing')
        test(net, test_loader, criterion, epoch)
        if epoch % save_step == 0:
            torch.save(net.state_dict(), args.model_root)