import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import os
import h5py
import time
from butterfly import Butterfly
from scipy.stats import ortho_group
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser(description='Sketching for low-rank matrix decomposition')

parser.add_argument('--experiment', default=1, type=int, help='experiment ID')

parser.add_argument('--m', default=20, type=int, help='value of m')

parser.add_argument('--k', default=10, type=int, help='value of k')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

parser.add_argument('--premute', default="False", type=str, help='randomly premute the dataset or not (default: False)')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')

parser.add_argument('--dataset', default='CIFAR10', type=str,
                    help='dataset (default: (CIFAR10), other options are HS-SOD')


args = parser.parse_args()

print (args)


args_data = args.dataset
args_lr = args.lr
args_epochs = args.epochs

experiment = args.experiment


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('using {} device'.format(device))
criterion= nn.MSELoss()
A_train=[]
A_test=[]

N = 400
scale = 10
m = args.m
k = args.k
pics = [1,2,6,8,9,10,12,14,15,16,17,18,19,20,21,22,24,26,27,28,29,31,32,33,34,36,37,38,40,41,42,43,44,45,46,47,50,51]
#inds = [i for i in range(81) if np.random.random()<0.8 ]
inds = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 26, 27, 28, 30, 32, 33, 35, 36, 38, 39, 40, 41, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 70, 71, 73, 74, 75, 76, 77, 79]


def get_CIFAR10_data(n_train,n_test):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trainset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,])
    , download=True)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=1, shuffle=True)    
    trainloader_n = torch.utils.data.Subset(train_loader.dataset, (list(range(n_train))))   
    valset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)
    val_loader_n = torch.utils.data.Subset(val_loader.dataset, (list(range(n_test))))
    lst_train =  list(trainloader_n)
    lst_test =  list(val_loader_n)
    data_mat = [lst_train[i][0][0] for i in range (n_train)]
    test_mat = [lst_test[i][0][0] for i in range (n_test)]
    A_train=[]
    A_test=[]
    for i in range(len(data_mat)):
        As = data_mat[i]
        U, S, V = As.svd()
        div = abs(S[0].item())
        if div < 1e-10:
            div = 1
            print("Catch!")
        div /= scale
        A_train.append(As/div)
    for i in range(len(test_mat)):
        As = test_mat[i]
        U, S, V = As.svd()
        div = abs(S[0].item())
        if div < 1e-10:
            div = 1
            print("Catch!")
        div /= scale
        A_test.append(As/div)
    return A_train,A_test
    

def get_hyper_data(N):
    A_train=[]
    A_test=[]
    for i in range(1,N):
        print(i)
        fname = 'HS-SOD/hyperspectral/'+str(pics[i]).zfill(4)+'.mat'
        f = h5py.File(fname, 'r')
        AList=f['hypercube'][:]
        for j in range(AList.shape[0]):
            As = torch.from_numpy(AList[j]).view(AList[j].shape[0], -1).float()
            U, S, V = As.svd()
            div = abs(S[0].item())
            if div < 1e-10:
                div = 1
                print("Catch!")
            div /= scale
            if j in inds:
                A_train.append(As/div)
            else:
                A_test.append(As/div)
    return A_train,A_test

def premute_data(trainset,testset):
    perm = np.random.permutation(d)
    rng = list(range(d))
    newtrain, newtest = [],[]
    for pic in trainset:
        npic = pic.numpy()
        npic[rng] = npic[perm]
        newtrain.append(torch.tensor(npic).float().to(device))
    for pic in testset:
        npic = pic.numpy()
        npic[rng] = npic[perm]
        newtest.append(torch.tensor(npic).float().to(device))
    return newtrain,newtest
    
def get_random_sparse_S(m,d):
    S = np.zeros((m,d))
    h = np.random.randint(0,m,d)
    signs = np.random.randint(-1,1,d) * 2 + 1
    for i in range (len(h)):
        S[h[i]][i] = signs[i]
    return S

def get_random_dense_S(m,d):
    return ortho_group.rvs(d)[:m] 

def calc_opt_loss(test_set,k):
    loss = 0
    for img in test_set:
        img = img.numpy()
        U,s,V = np.linalg.svd(img, full_matrices=False)
        Zpca = np.dot(img, V.transpose())
        Rpca = np.dot(Zpca[:,:k], V[:k,:])   # reconstruction
        err = np.linalg.norm(img-Rpca)
        loss+=err
    return loss/len(test_set)

def calc_sketch_loss(S,test_set,k):
    loss = 0
    for img in test_set:
        img = img.numpy()
        new_img = S@img
        U,s,Vt = np.linalg.svd(new_img, full_matrices=False)
        Zpca = np.dot(img, Vt.transpose())
        Unew,snew,Vtnew = np.linalg.svd(Zpca, full_matrices=False)
        Zpcanew = np.dot(Zpca, Vtnew.transpose())
        Rpcanew = np.dot(Zpcanew[:,:k], Vtnew[:k,:])   # reconstruction
        res = Rpcanew@Vt
        err = np.linalg.norm(img-res)
        loss+= err
    return (loss - bestPossible(test_set,k))/len(test_set)

def bestPossible(eval_list,k):
    totLoss = 0
    for A in eval_list:
        print(".",end="")
        AM=A.to(device)
        U, S, V = AM.svd()
        ans = U[:, :k].mm(torch.diag(S[:k]).to(device)).mm(V.t()[:k])
        # totLoss += torch.norm(ans - AM) ** 2
        totLoss += torch.norm(ans - AM)
    return totLoss

def mysvd(init_A,k):
    if k>min(init_A.size(0),init_A.size(1)):
        k=min(init_A.size(0),init_A.size(1))
    d=init_A.size(1)
    x=[torch.Tensor(d).uniform_() for i in range(k)]
    for i in range(k):
        x[i]=x[i].to(device)
        x[i].requires_grad=False
    def perStep(x,A):
        x2=A.t().mv(A.mv(x))
        x3=x2.div(torch.norm(x2))
        return x3
    U=[]
    S=[]
    V=[]
    Alist=[init_A]
    for kstep in range(k): #pick top k eigenvalues
        cur_list=[x[kstep]]   #current history
        for j in range(300):  #steps
            cur_list.append(perStep(cur_list[-1],Alist[-1]))  #works on cur_list
        V.append((cur_list[-1]/torch.norm(cur_list[-1])).view(1,cur_list[-1].size(0)))
        S.append((torch.norm(Alist[-1].mv(V[-1].view(-1)))).view(1))
        U.append((Alist[-1].mv(V[-1].view(-1))/S[-1]).view(1,Alist[-1].size(0)))
        Alist.append(Alist[-1]-torch.ger(Alist[-1].mv(cur_list[-1]), cur_list[-1]))
    return torch.cat(U,0).t(),torch.cat(S,0),torch.cat(V,0).t()

def evaluate(eval_list,sketch_vector, sketch_value,m,k,n,d):  # evaluate the test/train performance
    totLoss = 0
    count = 0
    for A in eval_list:
        AM=A.to(device)
        SA = torch.Tensor(m, d).fill_(0).to(device)
        for i in range(n):  # A has this many rows, not mapped yet
            mapR = sketch_vector[i]  # row is mapped to this row in the sketch
            SA[mapR] += AM[i] * sketch_value[i]  # remember: times the weight
        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
    return totLoss

def sparse_train_SVD(model, optimizer, n_epochs,print_ratio):
    for epoch in range(1, n_epochs + 1):
        A = A_train[int(torch.randint(N_train, [1]).item())]
        img = A.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        grad = list(model.parameters())[0].grad
        grad[masks] = 0
        optimizer.step()
        if epoch%print_ratio == 1:
            print (loss.item(),epoch)
        print('Finished Training')

        
if args_data == 'HS-SOD':
    n = 1024
    d = 768
    A_train,A_test = get_hyper_data(25)
    N_train=25
    N_test=5
    A_train = A_train[:N_train]
    A_test = A_test[:N_test]
    best_train = bestPossible(A_train,k).tolist()/N_train
    best_test = bestPossible(A_test,k).tolist()/N_test

else:
    print ('Using CIFAR-10 Dataset')
    n = 32
    d = 32
    A_train, A_test = get_CIFAR10_data(N,100)
    N_train=len(A_train)
    N_test=len(A_test)
    best_train = bestPossible(A_train,k).tolist()/N_train
    best_test = bestPossible(A_test,k).tolist()/N_test


if args.premute == 'True':

    A_train, A_test = premute_data(A_train,A_test)

print ('best train loss:', best_train)
print ('best test loss:', best_test)

class butterflyAutoEncode(nn.Module):
    def __init__(self,k=10,m=20,d=1024):
        super(butterflyAutoEncode, self).__init__()
        self.B = nn.Linear(d,m, bias=False)
    def forward(self, x):
        xx = torch.transpose(self.B(torch.transpose(x,0,1)),0,1)
        U2, Sigma2, V2 = mysvd(xx, xx.size(1))
        AU = x.mm(V2)
        U3, Sigma3, V3 = mysvd(AU, k)
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        return ans
def train(model, optimizer, n_epochs):
    for epoch in range(1, n_epochs + 1):
        A = A_train[int(torch.randint(N_train, [1]).item())]
        img = A.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        # update running training loss
        print (loss.item(),epoch)
    print('Finished Training')
def test(model,eval_set,k,optloss):  
    loss = 0
    B = model.B
    for A in eval_set:
        AM= A.to(device)
        SA= torch.transpose(B(torch.transpose(AM,0,1)),0,1)
        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        loss += (torch.norm(ans - AM)).item()
    print('Finished Testing')
    print ("test loss is: " , (loss/len(eval_set))-optloss)
    return (loss/len(eval_set)) - optloss

if experiment == 1:

    print ('Running experiment 1')

    #Experiment 1: random trained, init_lr = 4, m=20,k=10 default

    rlt_dic={}
    print_freq=5
    num_iter = 50000
    start=time.time()
    cur_diff = []
    lr = 4
    steps = []
    losses_train = []
    losses_test = []
    sketch_vector = torch.randint(m, [n]).int()  # m*n
    sketch_vector.requires_grad = False
    sketch_value = ((torch.randint(2, [n]).float() - 0.5) * 2).cuda()
    sketch_value.requires_grad = False
    for bigstep in range(1,num_iter+1):
        if ((bigstep+1)%1000==0) and lr>1:
            lr=lr*0.3
        A = A_train[int(torch.randint(N_train, [1]).item())]
        AM = A.cuda()
        Ad=d
        An=n
        if bigstep % print_freq == 1:
            print(bigstep, '.')
            f_name ='m='+str(m)+'_k='+str(k)+'_iter=' + str(bigstep)+'_N='+str(N)+'_scale='+str(scale)
            rlt_dic[f_name] = (evaluate(A_train,sketch_vector,sketch_value,m,k,An,Ad),
                              evaluate(A_test,sketch_vector,sketch_value,m,k,An,Ad))
            print(f_name, rlt_dic[f_name][0]/N_train-best_train, rlt_dic[f_name][1]/N_test-best_test)
            steps.append(bigstep)
            losses_train.append(rlt_dic[f_name][0]/N_train-best_train)
            losses_test.append(rlt_dic[f_name][1]/N_test-best_test)
        SA = torch.Tensor(m, Ad).fill_(0).cuda()
        for i in range(n):  # A has this many rows, not mapped yet
            mapR = sketch_vector[i]  # row is mapped to this row in the sketch
            SA[mapR] += AM[i] * sketch_value[i]  # remember: times the weight
        SH = SA.detach()
        SH.requires_grad = True
        U2, Sigma2, V2 = mysvd(SH, SH.size(1))
        AU = AM.mm(V2)
        U3, Sigma3, V3 = mysvd(AU, k)
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).cuda()).mm(V3.t()[:k]).mm(V2.t())
        loss = torch.norm(ans - AM)
        loss.backward()
        if bigstep%10==0:
            print(loss.cpu().item(),loss.cpu().item()-best_train, end=",")
        for i in range(n):
            sketch_value[i] -=lr* torch.dot(SH.grad.data[int(sketch_vector[i]), :], AM[i, :])
        del SA, SH, U2, Sigma2, V2, AU, U3, Sigma3, V3, ans, loss, AM
        torch.cuda.empty_cache()

if experiment == 2:

    #Experiment 2: Butterfly trained, init_lr = 0.01, m=20,k=10 is the default
    
    steps_per_epoch = 50 
    total_steps = 50000
    model = butterflyAutoEncode(k=k,m=m,d=n)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01) # default lr, lr = 0.01 . 0.1. 0.5 
    model.to(device)
    optlosstrain = bestPossible(A_train,k)/N_train
    optlosstest = bestPossible(A_test,k)/N_test
    steps = []
    train_losses = []
    test_losses = []
    for i in range(1,total_steps,steps_per_epoch):
        print (i)
        train_losses.append(test(model,A_train,k,optlosstrain))
        test_losses.append(test(model,A_test,k,optlosstest))
        steps.append(i)
        train(model,optimizer,steps_per_epoch)

#Experiment 3: try 100 random sparse and 100 random dense and take the avg loss on test set, m=20, k=10 are default

if experiment == 3:
    n_times = 100
    loss_random_sparse = 0
    loss_random_dense = 0
    for i in range (n_times):
        S = get_random_sparse_S(m,n)
        loss_random_sparse += (calc_sketch_loss(S,A_test,k)/n_times)
        S = get_random_dense_S(m,n)
        loss_random_dense += (calc_sketch_loss(S,A_test,k)/n_times)



    print ('\n',loss_random_sparse)
    print ('',loss_random_dense)



#Experiment 4 setup
combos = [(10,10),(10,20),(10,40),(10,60),(10,80),(20,20),(20,40),(30,30),(60,60)]
combos_rest = [(10,10),(10,40),(10,60),(10,80),(20,20),(20,40),(30,30),(60,60)]
new_combos = [(1,5),(1,10),(1,20),(1,40),(1,60),(1,80)]

#Experiment 4: try 5 random sparse and 5 random dense and take the avg loss on test set, for all combos

if experiment == 4:
    combo_losses_sparse_dense = []
    n_times = 5
    for combo in combos:
        k,m = combo
        loss_random_sparse = 0
        loss_random_dense = 0
        for i in range (n_times):
            S = get_random_sparse_S(m,n)
            loss_random_sparse += (calc_sketch_loss(S,A_test,k)/n_times)
            S = get_random_dense_S(m,n)
            loss_random_dense += (calc_sketch_loss(S,A_test,k)/n_times)
        combo_losses_sparse_dense.append((loss_random_sparse,loss_random_dense))
        print ('finished combo' , combo)

    print (combo_losses_sparse_dense)

#Experiment 5: try random sparse, for all combos

if experiment == 5:
    combo_losses_test = []
    for combo in new_combos:
        best_acc = 100000
        k,m = combo
        best_train = bestPossible(A_train,k).tolist()/N_train
        best_test = bestPossible(A_test,k).tolist()/N_test
        rlt_dic={}
        print_freq=5
        num_iter = 752
        start=time.time()
        cur_diff = []
        lr = 4
        sketch_vector = torch.randint(m, [n]).int()  # m*n
        sketch_vector.requires_grad = False
        sketch_value = ((torch.randint(2, [n]).float() - 0.5) * 2).cuda()
        sketch_value.requires_grad = False
        for bigstep in range(1,num_iter+1):
            if ((bigstep+1)%1000==0) and lr>1:
                lr=lr*0.3
            A = A_train[int(torch.randint(N_train, [1]).item())]
            AM = A.cuda()
            Ad=d
            An=n
            if bigstep % print_freq == 1:
                print(bigstep, '.')
                f_name ='m='+str(m)+'_k='+str(k)+'_iter=' + str(bigstep)+'_N='+str(3)+'_scale='+str(scale)
                rlt_dic[f_name] = (evaluate(A_train,sketch_vector,sketch_value,m,k,An,Ad),
                                  evaluate(A_test,sketch_vector,sketch_value,m,k,An,Ad))
                print(f_name, rlt_dic[f_name][0]/N_train-best_train, rlt_dic[f_name][1]/N_test-best_test)
                best_acc = min(best_acc,rlt_dic[f_name][1]/N_test-best_test)
            SA = torch.Tensor(m, Ad).fill_(0).cuda()
            for i in range(n):  # A has this many rows, not mapped yet
                mapR = sketch_vector[i]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[i]  # remember: times the weight
            SH = SA.detach()
            SH.requires_grad = True
            U2, Sigma2, V2 = mysvd(SH, SH.size(1))
            AU = AM.mm(V2)
            U3, Sigma3, V3 = mysvd(AU, k)
            ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).cuda()).mm(V3.t()[:k]).mm(V2.t())
            loss = torch.norm(ans - AM)
            loss.backward()
            if bigstep%10==0:
                print(loss.cpu().item(),loss.cpu().item()-best_train, end=",")
            for i in range(n):
                sketch_value[i] -=lr* torch.dot(SH.grad.data[int(sketch_vector[i]), :], AM[i, :])
            del SA, SH, U2, Sigma2, V2, AU, U3, Sigma3, V3, ans, loss, AM
            torch.cuda.empty_cache()
        combo_losses_test.append(best_acc)

    print (combo_losses_test)


new_combos = [(1,5),(1,10),(1,20),(1,40),(1,60),(1,80)]

#Experiment 6: try random Butterfly, for all combos


if experiment == 6:

    combo_losses_test = []
    for combo in new_combos:
        best_acc = 100000
        k,m = combo
        steps_per_epoch = 5
        total_steps = 852 
        model = butterflyAutoEncode(k=k,m=m,d=n)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.01)  
        model.to(device)
        optlosstrain = bestPossible(A_train,k)/N_train
        optlosstest = bestPossible(A_test,k)/N_test
        for i in range(1,total_steps,steps_per_epoch):
            print (i)
            best_acc = min(best_acc,test(model,A_test,k,optlosstest).item())
            train(model,optimizer,steps_per_epoch)
        combo_losses_test.append(best_acc)

    print (combo_losses_test)
