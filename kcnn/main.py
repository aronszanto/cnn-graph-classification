import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.autograd import Variable
import numpy as np
from utils import compute_nystrom,create_train_test_loaders
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from model import CNN
import pickle as pkl

# Dataset
# data_file = "IMDB-BINARY"

# Community detection method
community_detection = "louvain"

# Hyper Parameters
dim = 200
batch_size = 32
num_epochs = 100
num_filters = 256
hidden_size = 128
learning_rate = 0.001



# unlabeled_data_files = ["IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K", "COLLAB", "SYNTHETIC"]
# if data_file in unlabeled_data_files:
use_node_labels = False
from graph_kernels import sp_kernel, wl_kernel
# else:
#     use_node_labels = True
#     from graph_kernels_labeled import sp_kernel, wl_kernel

# Choose kernels
kernels=[wl_kernel]
num_kernels = len(kernels)
ds_name = "bias"
seed = 42
print("Computing feature maps...")
Q, subgraphs, labels,shapes = compute_nystrom(ds_name, use_node_labels, dim, community_detection, kernels, seed)
print("Finished feature maps")
M=np.zeros((shapes[0],shapes[1],len(kernels)))
for idx,k in enumerate(kernels):
    M[:,:,idx]=Q[idx]

Q=M

# Binarize labels
le = LabelEncoder()
y = le.fit_transform(labels)
print("Building vocabulary")
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in subgraphs])
x = np.zeros((len(subgraphs), max_document_length), dtype=np.int32)
for i in range(len(subgraphs)):
    # print(i, "/", len(subgraphs))
    communities = subgraphs[i].split()
    for j in range(len(communities)):
        x[i,j] = int(communities[j])

pkl.dump(x, open('x_news.pkl', 'wb'))


kf = KFold(n_splits=10, random_state=None)
kf.shuffle=True
train_accs = []
test_accs = []
it = 0

print("Starting cross-validation...")

for train_index, test_index in kf.split(x):
    it += 1
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_loader, test_loader = create_train_test_loaders(Q, x_train, x_test, y_train, y_test, batch_size)

    cnn = CNN(input_size=num_filters, hidden_size=hidden_size, num_classes=np.unique(y).size, dim=dim, num_kernels=num_kernels, max_document_length=max_document_length)
    if torch.cuda.is_available():
        cnn.cuda()

    # Loss and Optimizer
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (graphs, labels) in enumerate(train_loader):
            graphs = Variable(graphs)
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = cnn(graphs)
            if torch.cuda.is_available():
                loss = criterion(outputs, labels.cuda())
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        if epoch % 10 == 0:
            print("Epoch %i: Loss = %.2f" % (epoch, total_loss))

    # Test the Model
    cnn.eval()
    correct = 0
    total = 0
    for graphs, labels in test_loader:
        graphs = Variable(graphs)
        outputs = cnn(graphs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted == labels.cuda()).sum()
        else:
            correct += (predicted == labels).sum()

    test_acc = (100 * correct / total)
    test_accs.append(test_acc)

    for graphs, labels in train_loader:
        graphs = Variable(graphs)
        outputs = cnn(graphs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted == labels.cuda()).sum()
        else:
            correct += (predicted == labels).sum()

    train_acc = (100 * correct / total)
    train_accs.append(train_acc)
    print("Accuracies at iteration "+ str(it) +": \n\t- Train: " + str(train_acc) + "\n\t- Test: " + str(test_acc))
    del train_loader
    del test_loader

print("Average accuracies:\n\t- Train: " + str(np.mean(train_accs)) + "\n\t- Test: " + str(np.mean(test_accs)))
