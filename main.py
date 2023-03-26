import torch 
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import pickle
import os 
import gc
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class block(nn.Module):
    def __init__(self,in_channels,out_channels,identity_down_sample = None,stride=1) -> None:
        super(block,self).__init__()
        # Number of channels is 4 times higher than what the input is
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_down_sample = identity_down_sample

    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x= self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x) 
        if self.identity_down_sample is not None:
            identity = self.identity_down_sample(identity)
        x+=identity
        x = self.relu(x)
        return x



class ResNet(nn.Module): 
    def __init__(self,block,layers,image_channels,num_classes) -> None:
        super(ResNet,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels,64,kernel_size=7,stride =2,padding=3)
        self.bn1 = nn.BatchNorm2d(64) 
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size =3 , stride =2,padding= 1)

        #ResNet Layers 
        self.layer1 = self._build_layer(block,layers[0],out_channels=64,stride=1)
        self.layer2 = self._build_layer(block,layers[1],out_channels=128,stride=2)
        self.layer3 = self._build_layer(block,layers[2],out_channels=256,stride=2)
        self.layer4 = self._build_layer(block,layers[3],out_channels=512,stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)
        # for this apllication , we may need to build an another layer of regression layer to predict the age from the final fully connected layer
        self.reg = nn.Linear(num_classes,1) 
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        #this is where the actual ResNet ends. After this doing simple linear regression to get the final age 
        x = self.reg(x)
        return x
    def _build_layer(self,block,num_residual_blocks,out_channels,stride):
        identity_down_sample = None
        layers = []
        # when are we doing the identity down sample?
        if stride!=1 or self.in_channels!= out_channels*4:
            identity_down_sample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels*4,kernel_size =1,stride = stride),
                                                    nn.BatchNorm2d(out_channels*4))
        layers.append(block(self.in_channels,out_channels,identity_down_sample,stride))
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks-1): # we are already done with one layer , that is why we are doing one less 
            layers.append(block(self.in_channels,out_channels)) #256 -> 64, 64*4 ->256  again 

        
        return(nn.Sequential(*layers))

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

def ResNet50(img_channels=1,num_classes=1000):

    return ResNet(block,[3,4,6,3],img_channels,num_classes)

# def ResNet101(img_channels =3,num_classes =1000):
#     return ResNet(block,[3,4,23,3],img_channels,num_classes)
# def ResNet152(img_channels =3,num_classes =1000):
#     return ResNet(block,[3,8,36,3],img_channels,num_classes)

def split_data(X,y,train_split=0.2,validation_split=0.1):
    print("Splitting the data....!")
    # randomize the data
    allIdxs = np.arange(X.shape[0]) 
    Idxs = np.random.permutation(allIdxs) #random indices for the train data 
    # select the 1st split  of the indices for thr train data and rest for test data 
    train_split = 1-train_split-validation_split
    train_part = Idxs[:int(len(Idxs)*train_split)]
    validation_part = Idxs[int(len(Idxs)*train_split):int(len(Idxs)*(train_split+validation_split))]
    test_part = Idxs[int(len(Idxs)*(train_split+validation_split)):]


    def convert_tensor(data,ctype="X"):
            # Define the transformation pipeline
        transform_X = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1), # keeping input channels same as gray image scale. change this to 3 to match rgb scale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        data_conv =[]
        for img in data:
            img = transform_X(img)
            data_conv.append(img)
        return torch.stack(data_conv).float()
    y= torch.from_numpy(y).float()
    y = y.reshape((-1,1))
    X_tr = convert_tensor(X[train_part])
    ytr = y[train_part]
    X_va = convert_tensor(X[validation_part])
    yva = y[validation_part]
    X_te = convert_tensor(X[test_part])
    yte = y[test_part]
    return X_tr,ytr,X_va,yva,X_te,yte

def train_model(model,criterion,optimizer,X_tr,ytr,X_va,yva,epochs,batch_size):
    for epoch in range(epochs):
        n =0 
        while n<len(ytr):
            #split the batch
            X = X_tr[n:n+batch_size,:]
            y = ytr[n:n+batch_size,:]
            output = model(X)
            loss = criterion(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()
            n+=batch_size
        print ('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
        output = model(X_va)
        loss = criterion(output,yva)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        gc.collect()
        print ('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))


    return
def test_model(model,criterion,optimizer,X_te,yte):
    output = model(X_te)
    loss = criterion(output,yte)
    torch.cuda.empty_cache()
    gc.collect()
    print ('Test Loss: {:.4f}'.format(loss.item()))
    return
def main():
    # we do not need to split the data each time
    # check if it already available
    # else reload the data and split
    # if os.path.exists("data.pickle"):
    #     print("Data file exists...!")
    #     with open('data.pickle', 'rb') as file:
    #         X_tr,ytr,X_va,yva,X_te,yte = pickle.load(file)
    # else:
    #     X = np.load("facesAndAges/faces.npy")
    #     y = np.load("facesAndAges/ages.npy") 
    #     A = split_data(X,y)
    #     X_tr,ytr,X_va,yva,X_te,yte  = A
    #     with open('data.pickle', 'wb') as handle:
    #         pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
    X = np.load("facesAndAges/faces.npy")
    y = np.load("facesAndAges/ages.npy") 
    X_tr,ytr,X_va,yva,X_te,yte  = split_data(X,y)
    print(ytr)
    #hyper parameters 
    epochs = 20
    batch_size = 512
    learning_rate = 0.01 
    print("loading model....!")
    model = ResNet50()#.to(device)
    # loss function 
    criterion = RMSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  
    print("Training the model...!")
    train_model(model,criterion,optimizer,X_tr,ytr,X_va,yva,epochs,batch_size)
    # print("Computing output....!")
    # output = model(X_va)
    # # print(output)
    # # print(yva)
    # loss = criterion(output,yva)
    # print(loss)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # torch.cuda.empty_cache()
    # print("Computing output again....!")
    # output = model(X_va)
    # # print(output)
    # # print(yva)
    # loss = criterion(output,yva)
    # print(loss)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # torch.cuda.empty_cache()






    
    # y = mdoel(X_tr)
    # print(y.shape)

    pass
if __name__ == "__main__":
    main()