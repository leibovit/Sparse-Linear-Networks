import sklearn
from sklearn import datasets as sdatasets
from scipy.stats import ortho_group
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
import os
from butterfly import Butterfly
import argparse


# Globals

criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d = 1024
logd = 10
print ('Using {} device'.format(device))


# Functions

def getFacesData():
    H = sdatasets.fetch_olivetti_faces(
        data_home=None, shuffle=False, random_state=0,
        download_if_missing=True).data
    size = H.shape[0]
    muH = np.mean(H, axis=0)
    H -= muH
    for i in range(size):
        H[i] /= np.linalg.norm(H[i])
    return H

def getLowRankData(dim = 1024 ,size = 1024,rank = 32):
    
    ort_vecs = ortho_group.rvs(dim)
    basis = ort_vecs[:rank]
    rand_mat = np.random.normal(0, 1, size * rank).reshape(size, rank)
    span = rand_mat @ basis
    mu = np.mean(span, axis=0)
    span -= mu
    for i in range(size):
        span[i] /= np.linalg.norm(span[i])
        
    return span

def getMnistData(size=1024):
    transform = transforms.ToTensor()
    # load the training and test datasets
    train_data = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    # prepare data loaders
    data_mat = train_data.data
    numpy_data = data_mat.numpy()
    padded_data = np.pad(numpy_data, ((0, 0), (2, 2), (2, 2)), 'constant')
    padded_data = padded_data.astype(float)
    padded_data = padded_data[:size] / 255
    X = padded_data.reshape(size, 1024)
    mu = np.mean(X, axis=0)
    X -= mu
    for i in range(size):
        X[i] /= np.linalg.norm(X[i])
    return X


def getTorchDataLoader(Dataset, batch_size=1024):
    tensor = torch.stack([torch.Tensor(i) for i in Dataset])  # transform to torch tensors
    my_dataset = torch.utils.data.TensorDataset(tensor, tensor)  # create your datset
    return torch.utils.data.DataLoader(my_dataset, batch_size)

def get_PCA_loss(Data):
    print ('Calculating PCA loss (optimal)')
    print()
    dim = Data.shape[1]
    U, s, V = np.linalg.svd(Data, full_matrices=False)
    ks, losses = [], []
    for k in range(1, dim, 24):
        Zpca = np.dot(Data, V.transpose())
        Rpca = np.dot(Zpca[:, :k], V[:k, :])  # reconstruction
        err = np.sum((Data - Rpca) ** 2) / Data.shape[0] / Data.shape[1]
        print('PCA  ' +str(k) +  ' PCs: ' + str(round(err,5)));
        ks.append(k)
        losses.append(err)
        if err < 1e-7:
            return ks,losses
    return ks, losses


def get_ort_vectors(dim):
    return ortho_group.rvs(dim)

def get_JL_loss(Data, ort_vecs):
    print ('Calculating JL loss')
    print()
    dim = Data.shape[1]
    # ort_vecs = get_ort_vectors(dim)
    ks_JL, losses_JL = [], []
    for k in range(1, dim + 1, 24):
        y = ort_vecs[:k]
        err = np.sum((Data - (Data @ y.transpose() @ y)) ** 2) / Data.shape[0] / Data.shape[1]
        print('random projection error with  ' +str(k) +  ' dimensions: ' + str(round(err,9)));
        ks_JL.append(k)
        losses_JL.append(err)
        if err < 1e-7:
            return ks,losses
    return ks_JL, losses_JL

def get_PCA_loss_lim(Data,lim):
    dim = Data.shape[1]
    U,s,V = np.linalg.svd(Data, full_matrices=False)
    ks, losses = [],[]
    for k in range (0,lim+1,1):
        Zpca = np.dot(Data, V.transpose())
        Rpca = np.dot(Zpca[:,:k], V[:k,:])   # reconstruction
        err = np.sum((Data-Rpca)**2)/Data.shape[0]/Data.shape[1]
        print('PCA  ' +str(k) +  ' PCs: ' + str(round(err,12)));
        ks.append(k)
        losses.append(err)
    return ks,losses

def get_JL_loss2(Data,ort_vecs,lim):
    #ort_vecs = get_ort_vectors(dim)
    ks_JL, losses_JL = [],[]
    for k in range (0,lim+1,1):
        kp = k*(int(np.log2(k+1))+1)
        y = ort_vecs[:kp]
        newdata = y@Data
        U,s,V = np.linalg.svd(newdata, full_matrices=False)
        Zpca = np.dot(Data, V.transpose())
        Rpca = np.dot(Zpca[:,:k], V[:k,:])   # reconstruction
        err = (np.linalg.norm(Data-Rpca)**2)#/Data.shape[0]/Data.shape[1]
        print('random projection error with  ' +str(k) +  ' dimensions: ' + str(round(err,9)));
        ks_JL.append(k)
        losses_JL.append(err)
    return ks_JL,losses_JL

def plot_graphs(ks_lists, losses_lists, DatasetName):
    colors = ['-b', '-r', '-g']
    labels = ['PCA reconstruction error', 'JL reconstruction error', 'Sparse Autoencoder reconstruction error']
    for ks, losses, color, label in zip(ks_lists, losses_lists, colors, labels):
        plt.plot(ks, losses, color, label=label)
        plt.legend(loc="upper right")
    plt.title("Losses comparison on " + DatasetName + " Dataset")
    plt.xlabel("Dimension")
    plt.ylabel("Loss")

def MNIST_test():
    
    print ('Running MNIST test')
    print()
    X = getMnistData()
    dim = X.shape[1]
    ks_lists = []
    losses_lists = []
    ks, losses = get_PCA_loss(X)
    JL_ks, JL_losses = get_JL_loss(X, ort_vecs_1024)
    ks_lists.append(ks)
    ks_lists.append(JL_ks)
    losses_lists.append(losses)
    losses_lists.append(JL_losses)
    plot_graphs(ks_lists, losses_lists, "MNIST")
    return ks_lists,losses_lists


def Faces_test():
    print ('Running Olivetti faces test')
    print()
    X = getFacesData()
    dim = X.shape[1]
    ks_lists = []
    losses_lists = []
    ks, losses = get_PCA_loss(X)
    JL_ks, JL_losses = get_JL_loss(X, ort_vecs_4096)
    ks_lists.append(ks)
    ks_lists.append(JL_ks)
    losses_lists.append(losses)
    losses_lists.append(JL_losses)
    plot_graphs(ks_lists, losses_lists, "Faces")
    return ks_lists,losses_lists
    
def LowRank_test(k):
    X = getLowRankData(1024,1024,k)
    dim = X.shape[1]
    ks_lists = []
    losses_lists = []
    ks , losses = get_PCA_loss_lim(X,k)
    JL_ks, JL_losses = get_JL_loss2(X,ort_vecs_1024,k)
    ks_lists.append(ks)
    ks_lists.append(JL_ks)
    losses_lists.append(losses)
    losses_lists.append(JL_losses)
    plot_graphs(ks_lists,losses_lists , "Low Rank")

    return ks_lists,losses_lists


def Hyper_test():
    X = image
    dim = X.shape[1]
    ks_lists = []
    losses_lists = []
    ks , losses = get_PCA_loss_lim(X,257)
    #ort_vecs_64 = get_ort_vectors(64)
    JL_ks, JL_losses = get_JL_loss2(X,ort_vecs_799,257)
    #ks_sparse = read_loss('faces_all_ks.txt')
    #losses_sparse = read_loss('losses_faces_all_ks.txt')
    ks_lists.append(ks)
    ks_lists.append(JL_ks)
    #ks_lists.append(ks_sparse)
    losses_lists.append(losses)
    losses_lists.append(JL_losses)
    #losses_lists.append(losses_sparse)
    plot_graphs(ks_lists,losses_lists)
    
def save_loss(loss,path):
    with open(path, 'a') as f:
        f.write(str(loss))
        f.write('\n')

def read_loss(path):
    lst= []
    with open(path, 'r') as f:
        lst = f.readlines()
    losses = [float((x.split('\n'))[0]) for x in new_lossses]
    return losses

def get_weights_and_masks(logd):
    weights,masks = [],[]
    d=2**logd
    for t in range(logd):
        A = torch.zeros((d, d))
        B = torch.ones((d,d),dtype=torch.bool)
        for i in range(d):
            for j in range(d):
                if (i ^ j == 0 or i ^ j == 2 ** t):
                    A[i][j] =  1/np.sqrt(2)
                    B[i][j] = 0
                    if (i^j == 0 and (i%(2**(t+1)))>=2**t):
                        A[i][i]*=-1
        weights.append(A.clone())
        masks.append(B.clone())
    return weights, masks


def get_weights_and_masks_ort(logd):
    weights,masks = [],[]
    d=2**logd
    for t in range(logd):
        A = torch.zeros((d, d))
        B = torch.ones((d,d),dtype=torch.bool)
        offset = 2**t
        for i in range(d):
            for j in range(d):
                if (i == j):
                    B[i][j] = 0
                    if (i^j == 0 and (i%(2**(t+1)))>=2**t): #second item
                        A[i][j] = -A[i-offset][j-offset].item()
                    else:
                        A[i][j] = np.random.uniform(0,1,1)[0]
                if (i ^ j == 2 ** t):
                    B[i][j] = 0
                    if j>i:
                        A[i][j] = np.sqrt(1- (A[i][j-offset].item())**2)
                    else:
                        A[i][j] = np.sqrt(1- (A[i-offset][j].item())**2)
        weights.append(A.clone())
        masks.append(B.clone())
    return weights, masks

class Autoencoder128(nn.Module):
    def __init__(self, k):
        super(Autoencoder128, self).__init__()
        self.fc1 = nn.Linear(d, 512, bias=False)
        self.fc2 = nn.Linear(512, 128, bias=False)
        self.fc3 = nn.Linear(128, k, bias=False)
        self.fc4 = nn.Linear(k, 128 , bias=False)
        self.fc5 = nn.Linear(128, 512, bias=False)
        self.fc6 = nn.Linear(512,d, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, k):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(d, k, bias=False)
        self.fc2 = nn.Linear(k,d, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class SparseAutoEncoderMNIST(nn.Module):
    def __init__(self, k,batch_size):
        super(SparseAutoEncoderMNIST, self).__init__()
        self.k = k
        self.batch_size = batch_size
        self.perm = (torch.randperm(d))[:self.k]
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.fc3 = nn.Linear(d, d, bias=False)
        self.fc4 = nn.Linear(d, d, bias=False)
        self.fc5 = nn.Linear(d, d, bias=False)
        self.fc6 = nn.Linear(d, d, bias=False)
        self.fc7 = nn.Linear(d, d, bias=False)
        self.fc8 = nn.Linear(d, d, bias=False)
        self.fc9 = nn.Linear(d, d, bias=False)
        self.fc10 = nn.Linear(d, d, bias=False)
        self.fc_decode = nn.Linear(self.k, d, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(weights[0])
            self.fc2.weight.copy_(weights[1])
            self.fc3.weight.copy_(weights[2])
            self.fc4.weight.copy_(weights[3])
            self.fc5.weight.copy_(weights[4])
            self.fc6.weight.copy_(weights[5])
            self.fc7.weight.copy_(weights[6])
            self.fc8.weight.copy_(weights[7])
            self.fc9.weight.copy_(weights[8])
            self.fc10.weight.copy_(weights[9])
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc9(x)
        x = self.fc10(x)
        y = torch.zeros(self.batch_size, self.k).to(device)
        for i in range(self.batch_size):
            y[i] = x[i][self.perm]
        x = self.fc_decode(y)
        return x
    
def sparse_train(model, optimizer, n_epochs,k,train_loader):
    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0
        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            images = data[0].to(device)
            # flatten images
            #images = images.view(images.size(0), -1)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs, images)
            loss.backward()
            # perform a single optimization step (parameter update)
            # here zero the masked gradients
            for i in range(logd):
                grad = list(model.parameters())[i].grad
                grad[masks[i]] = 0
            optimizer.step()
            # update running training loss
            train_loss += loss.item()
            # print avg training statistics
        train_loss = train_loss / len(train_loader)
        
        if epoch%100 ==1:
            print('Epoch: {} \tTraining Loss: {:.7f}'.format(
                epoch,
                train_loss
            ))
            print('finished epoch ', epoch, 'with k = ', k)
    print('Finished Training')

def test(model, train_loader):
    loss_cnt = 0
    with torch.no_grad():
        for data in train_loader:
            images = data[0].to(device)
            # images = images.view(images.size(0), -1)
            outputs = model(images)
            loss = criterion(outputs, images)
            loss_cnt += loss
        loss_cnt /= len(train_loader)
        loss = loss_cnt.item()
        print("loss = ", loss)
    return loss

def get_sparse_loss(dataset, datasetName, batch_size=8, n_epochs=1):
    dim = dataset.shape[1]
    train_loader = getTorchDataLoader(dataset, batch_size=batch_size)
    weights, masks = get_weights_and_masks(int(np.log2(dim)))
    ks = [1, 2, 4, 8, 16, 32] + list(range(50, 1002, 50)) + [1024]
    for k in ks:
        if (datasetName == 'MNIST'):
            model = SparseAutoEncoderMNIST(k=k, batch_size=batch_size)
        if (datasetName == 'Faces'):
            model = SparseAutoEncoderFaces(k=k, batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.to(device)
        sparse_train(model, optimizer, n_epochs, k, train_loader)
        loss = test(model, train_loader)
        save_loss(k, "1024k_50epochs.txt")
        save_loss(loss, "1024k_loss_50epochs.txt")
    return sparse_ks, sparse_losses

    dataset = getMnistData()
    get_sparse_loss(dataset, 'MNIST', batch_size=1, n_epochs=50)


class SparseAutoEncoder32MNIST(nn.Module):
    def __init__(self, k,batch_size):
        super(SparseAutoEncoder32MNIST, self).__init__()
        self.k = k
        self.K = 1024 // self.k
        self.batch_size = batch_size
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.fc3 = nn.Linear(d, d, bias=False)
        self.fc4 = nn.Linear(d, d, bias=False)
        self.fc5 = nn.Linear(d, d, bias=False)
        self.fc6 = nn.Linear(d, d, bias=False)
        self.fc7 = nn.Linear(d, d, bias=False)
        self.fc8 = nn.Linear(d, d, bias=False)
        self.fc9 = nn.Linear(d, d, bias=False)
        self.fc10 = nn.Linear(d, d, bias=False)
        self.fc_encode = nn.Linear(self.K,self.k,bias=False)
        self.fc_decode = nn.Linear(k, d, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(weights[0])
            self.fc2.weight.copy_(weights[1])
            self.fc3.weight.copy_(weights[2])
            self.fc4.weight.copy_(weights[3])
            self.fc5.weight.copy_(weights[4])
            self.fc6.weight.copy_(weights[5])
            self.fc7.weight.copy_(weights[6])
            self.fc8.weight.copy_(weights[7])
            self.fc9.weight.copy_(weights[8])
            self.fc10.weight.copy_(weights[9])
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)
        y = torch.zeros(self.batch_size, self.K).to(device)
        for i in range(self.batch_size):
            perm = (torch.randperm(x[i].size(0))[:self.K])
            perm = torch.sort(perm).values
            y[i] = x[i][perm]
        x = self.fc_encode(y)
        x = self.fc_decode(x)
        return x

class SparseAutoEncoderFaces(nn.Module):
    def __init__(self, k,batch_size):
        super(SparseAutoEncoderFaces, self).__init__()
        self.k = k
        self.batch_size = batch_size
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.fc3 = nn.Linear(d, d, bias=False)
        self.fc4 = nn.Linear(d, d, bias=False)
        self.fc5 = nn.Linear(d, d, bias=False)
        self.fc6 = nn.Linear(d, d, bias=False)
        self.fc7 = nn.Linear(d, d, bias=False)
        self.fc8 = nn.Linear(d, d, bias=False)
        self.fc9 = nn.Linear(d, d, bias=False)
        self.fc10 = nn.Linear(d, d, bias=False)
        self.fc11 = nn.Linear(d, d, bias=False)
        self.fc12 = nn.Linear(d, d, bias=False)
        self.fc_decode = nn.Linear(self.k, d, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(weights[0])
            self.fc2.weight.copy_(weights[1])
            self.fc3.weight.copy_(weights[2])
            self.fc4.weight.copy_(weights[3])
            self.fc5.weight.copy_(weights[4])
            self.fc6.weight.copy_(weights[5])
            self.fc7.weight.copy_(weights[6])
            self.fc8.weight.copy_(weights[7])
            self.fc9.weight.copy_(weights[8])
            self.fc10.weight.copy_(weights[9])
            self.fc11.weight.copy_(weights[10])
            self.fc12.weight.copy_(weights[11])

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)
        x = self.fc11(x)
        x = self.fc12(x)
        y = torch.zeros(self.batch_size, self.k).to(device)
        for i in range(self.batch_size):
            perm = (torch.randperm(x[i].size(0))[:self.k])
            perm = torch.sort(perm).values
            y[i] = x[i][perm]
        x = self.fc_decode(y)
        return x

class SparseAutoEncoder64Faces(nn.Module):
    def __init__(self, k,batch_Size):
        super(SparseAutoEncoder64Faces, self).__init__()
        self.k = k
        self.K = 1024 // self.k
        self.batch_size = batch_size
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.fc3 = nn.Linear(d, d, bias=False)
        self.fc4 = nn.Linear(d, d, bias=False)
        self.fc5 = nn.Linear(d, d, bias=False)
        self.fc6 = nn.Linear(d, d, bias=False)
        self.fc7 = nn.Linear(d, d, bias=False)
        self.fc8 = nn.Linear(d, d, bias=False)
        self.fc9 = nn.Linear(d, d, bias=False)
        self.fc10 = nn.Linear(d, d, bias=False)
        self.fc11 = nn.Linear(d, d, bias=False)
        self.fc12 = nn.Linear(d, d, bias=False)
        self.fc_encode = nn.Linear(self.K,self.k,bias=False)
        self.fc_decode = nn.Linear(k, d, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(weights[0])
            self.fc2.weight.copy_(weights[1])
            self.fc3.weight.copy_(weights[2])
            self.fc4.weight.copy_(weights[3])
            self.fc5.weight.copy_(weights[4])
            self.fc6.weight.copy_(weights[5])
            self.fc7.weight.copy_(weights[6])
            self.fc8.weight.copy_(weights[7])
            self.fc9.weight.copy_(weights[8])
            self.fc10.weight.copy_(weights[9])
            self.fc11.weight.copy_(weights[10])
            self.fc12.weight.copy_(weights[11])
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)
        x = self.fc11(x)
        x = self.fc12(x)
        y = torch.zeros(self.batch_size, self.K).to(device)
        for i in range(self.batch_size):
            perm = (torch.randperm(x[i].size(0))[:self.K])
            perm = torch.sort(perm).values
            y[i] = x[i][perm]
        x = self.fc_encode(y)
        x = self.fc_decode(x)
        return x

def train(model,optimizer,n_epochs,k,train_loader):

    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0
        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            images = data[0].to(device)
            # clear the gradients of all optimized variables
            #images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            # here zero the masked gradients
            optimizer.step()
            # update running training loss
            train_loss += loss.item()
            # print avg training statistics
        train_loss = train_loss / len(train_loader)
        if epoch%100 ==1:
            print('Epoch: {} \tTraining Loss: {:.7f}'.format(
                epoch,
                train_loss
            ))
            print('finished epoch ', epoch, 'with k = ', k)
    print('Finished Training')
    

def test(model,train_loader):
    loss_cnt = 0
    with torch.no_grad():
        for data in train_loader:
            images = data[0].to(device)
            #images = images.view(images.size(0), -1)
            outputs = model(images)
            loss = criterion(outputs, images)
            loss_cnt += loss
        loss_cnt /= len(train_loader)
        loss = loss_cnt.item()
        print("loss = ", loss)
    return loss

class butterflyAutoEncode(nn.Module):
    def __init__(self,k,d=1024):
        super(butterflyAutoEncode, self).__init__()
        self.k = k
        self.B = Butterfly(in_size=d, out_size=self.k, bias=False, complex=False,
                              tied_weight=False, increasing_stride=True, ortho_init=True)
        #self.B.twiddle.requires_grad = False
        
        self.fc_decode = nn.Linear(self.k, d, bias=True)
    def forward(self, x):
        x = self.B(x)
        x = self.fc_decode(x)
        return x

class butterflyAutoEncodeFre(nn.Module):
    def __init__(self,k,d=1024):
        super(butterflyAutoEncodeFre, self).__init__()
        self.k = k
        self.B = Butterfly(in_size=d, out_size=self.k, bias=False, complex=False,
                              tied_weight=False, increasing_stride=True, ortho_init=True)
        self.B.twiddle.requires_grad = False
        
        self.fc_decode = nn.Linear(self.k, d, bias=True)
    def forward(self, x):
        x = self.B(x)
        x = self.fc_decode(x)
        return x

class butterflyAutoEncodeFre2(nn.Module):
    def __init__(self,k,d=1024):
        super(butterflyAutoEncodeFre2, self).__init__()
        self.k = k
        self.K = (int(np.log(self.k))+2)*self.k
        self.B = Butterfly(in_size=d, out_size=self.K, bias=False, complex=False,
                              tied_weight=False, increasing_stride=True, ortho_init=True)
        self.B.twiddle.requires_grad = False
        self.encode = nn.Linear(self.K,self.k)
        self.fc_decode = nn.Linear(self.k, d)
    def forward(self, x):
        x = self.B(x)
        x = self.encode(x)
        x = self.fc_decode(x)
        return x


class butterflyAutoEncodeFre3(nn.Module):
    def __init__(self,k,d=1024):
        super(butterflyAutoEncodeFre3, self).__init__()
        self.k = k
        self.K = (int(np.log(self.k))+2)*self.k
        self.B = Butterfly(in_size=d, out_size=self.K, bias=False, complex=False,
                              tied_weight=False, increasing_stride=True, ortho_init=True)
        self.encode = nn.Linear(self.K,self.k)
        self.fc_decode = nn.Linear(self.k, d)
    def forward(self, x):
        x = self.B(x)
        x = self.encode(x)
        x = self.fc_decode(x)
        return x
    
    
def trainbf(model, optimizer, n_epochs,k,train_loader):
    epochs = []
    errs = []
    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0
        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            images = data[0].to(device)
            # flatten images
            #images = images.view(images.size(0), -1)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
                # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            optimizer.step()
            # update running training loss
            train_loss += loss.item()
        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        if epoch % 5 == 1:
            epochs.append(epoch)
            errs.append(train_loss)
            print (train_loss,epoch,k)
    print('Finished Training')
    return train_loss,epochs,errs




parser = argparse.ArgumentParser(description='Two-Phase learning')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

parser.add_argument('--epochs', default=5000, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
                                
parser.add_argument('--dataset', default='MNIST', type=str,
                    help='dataset (default: (MNIST), other options are Faces, Gaussian')




args = parser.parse_args()

print (args)



args_data = args.dataset

args_epochs = args.epochs



if args_data == 'MNIST':
    
    print ('Working on MNIST data')
    dataset = getMnistData(512)[0:512]
    train_loader = getTorchDataLoader(dataset, batch_size=512)
    ks = [1,2,4,8,16,32,64,128,228,300,400,512]
    ers_freeze = []
    err_with_init = []
    err_random_init = []
    epochs_freeze = []
    losses_epochs_freeze = []
    epochs_reg = []
    n_epochs_freeze = args_epochs
    n_epochs_reg = args_epochs

    for k in ks :
        print ('k = {}'.format(k))
        print()
        model = butterflyAutoEncodeFre2(k)
        rand_init_model =  butterflyAutoEncodeFre3(k)
        rand_init_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters())
        model.to(device)
        rand_init_optimizer = torch.optim.Adam(rand_init_model.parameters())
        rand_init_model.to(device)
        print ('froze init: \n')
        train(model,optimizer,n_epochs_freeze,k,train_loader)
        model.B.twiddle.requires_grad = True
        print ('optimal init: \n')
        train(model,optimizer,n_epochs_reg,k,train_loader)
        print ('rand init: \n')   
        train(rand_init_model,rand_init_optimizer,n_epochs_freeze,k,train_loader)



if args_data == 'Faces' or True:
    
    print ('Working on Faces data')
    mat = getFacesData()
    train_loader = getTorchDataLoader(mat,batch_size=4096)
    ks = [1,2,4,8,16,32,64,128,228,300,400,512]
    ers_freeze = []
    err_with_init = []
    err_random_init = []
    epochs_freeze = []
    losses_epochs_freeze = []
    epochs_reg = []
    n_epochs_freeze = args_epochs
    n_epochs_reg = args_epochs

    for k in ks :
        print ('k = {}'.format(k))
        print()
        model = butterflyAutoEncodeFre2(k,d=4096)
        rand_init_model =  butterflyAutoEncodeFre3(k,d=4096)
        rand_init_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters())
        model.to(device)
        rand_init_optimizer = torch.optim.Adam(rand_init_model.parameters())
        rand_init_model.to(device)
        print ('froze init: \n')
        train(model,optimizer,n_epochs_freeze,k,train_loader)
        model.B.twiddle.requires_grad = True
        print ('optimal init: \n')
        train(model,optimizer,n_epochs_reg,k,train_loader)
        print ('rand init: \n')   
        train(rand_init_model,rand_init_optimizer,n_epochs_freeze,k,train_loader) 



if args_data == 'Gaussian':
    
    print ('Working on Gaussian data')
    mat = get_ort_vectors(1024)
    train_loader = getTorchDataLoader(mat,batch_size=512)
    ks = [1,2,4,8,16,32,64,128,228,300,400,512]
    ers_freeze = []
    err_with_init = []
    err_random_init = []
    epochs_freeze = []
    losses_epochs_freeze = []
    epochs_reg = []
    n_epochs_freeze = args_epochs
    n_epochs_reg = args_epochs

    for k in ks :
        print ('k = {}'.format(k))
        print()
        model = butterflyAutoEncodeFre2(k)
        rand_init_model =  butterflyAutoEncodeFre3(k)
        rand_init_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters())
        model.to(device)
        rand_init_optimizer = torch.optim.Adam(rand_init_model.parameters())
        rand_init_model.to(device)
        print ('froze init: \n')
        train(model,optimizer,n_epochs_freeze,k,train_loader)
        model.B.twiddle.requires_grad = True
        print ('optimal init: \n')
        train(model,optimizer,n_epochs_reg,k,train_loader)
        print ('rand init: \n')   
        train(rand_init_model,rand_init_optimizer,n_epochs_freeze,k,train_loader)    

# python Two-phase\ learning.py --epochs 5000 --dataset 'MNIST' --batch-size 128 --lr 0.001
