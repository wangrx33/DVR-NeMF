from mimetypes import init
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
import numpy as np

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1,1],padding=[0,1,0],first=False) -> None:
        super(Bottleneck,self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride[0],padding=padding[0],bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding[1],bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels*4,kernel_size=1,stride=stride[2],padding=padding[2],bias=False),
            nn.BatchNorm2d(out_channels*4)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels*4)
            )
        # if stride[1] != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         # 卷积核为1 进行升降维
        #         # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
        #         nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride[1], bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )
    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet50(nn.Module):
    def __init__(self,Bottleneck, args) -> None:
        super(ResNet50, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv0 = nn.Sequential(
            nn.Conv3d(1,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # conv2
        self.conv2 = self._make_layer(Bottleneck,64,[[1,1,1]]*3,[[0,1,0]]*3)

        # conv3
        self.conv3 = self._make_layer(Bottleneck,128,[[1,2,1]] + [[1,1,1]]*3,[[0,1,0]]*4)

        # conv4
        self.conv4 = self._make_layer(Bottleneck,256,[[1,2,1]] + [[1,1,1]]*5,[[0,1,0]]*6)

        # conv5
        self.conv5 = self._make_layer(Bottleneck,512,[[1,2,1]] + [[1,1,1]]*2,[[0,1,0]]*3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, args.NumSlice)

    def _make_layer(self,block,out_channels,strides,paddings):
        layers = []
        # 用来判断是否为每个block层的第一层
        flag = True
        for i in range(0,len(strides)):
            layers.append(block(self.in_channels,out_channels,strides[i],paddings[i],first=flag))
            flag = False
            self.in_channels = out_channels * 4
            

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x) # b1, c64, 256, 140, 288
        out = out.view(-1, 64*int(np.ceil(x.shape[-3]/4)), int(np.ceil(x.shape[-2]/4)), int(np.ceil(x.shape[-1]/4)))   # b1, c64*64, h35, w72
        out = self.conv1(out)   # 35, 72
        out = self.conv2(out)   # 35, 72
        out = self.conv3(out)   # 18, 36
        out = self.conv4(out)   # 9, 18
        out = self.conv5(out)   # 2048, 5, 9

        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

class VGG19(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(1,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(4096,1024,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(1024,256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU()
        )
        # self.conv1=nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv4=nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3,padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.conv11= nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.conv12= nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
        self.conv13 = nn.Conv2d(512, 1024, kernel_size=3,padding=1)
        self.conv14 = nn.Conv2d(1024, 1024, kernel_size=3,padding=1)
        self.conv15 = nn.Conv2d(1024, 1024, kernel_size=3,padding=1)
        self.conv16 = nn.Conv2d(1024, 1024, kernel_size=3,padding=1)
        self.pool5=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(1024,2048)
        self.fc2=nn.Linear(2048,2048)
        self.fc3=nn.Linear(1024, args.NumSlice)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()
        self.prelu=nn.PReLU()
        # self.softmax=nn.Softmax()
        print("success")
    def forward(self,input):
        x = self.conv0(input)
        x = x.view(-1, 64*int(np.ceil(input.shape[-3]/4)), int(np.ceil(input.shape[-2]/4)), int(np.ceil(input.shape[-1]/4)))   # b1, c64*256/4, h140/4, w512/4
        x = self.conv1(x)   # 64, 35, 64

        x = self.relu(self.conv2(x))    # 64, 35, 64
        x = self.pool1(x)               # 64, 18, 32
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)               # 128, 9, 16
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.pool3(x)               # 256, 5, 8
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.pool4(x)               # 512, 3, 4
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))
        x = self.pool5(x)               # 1024, 2, 2
        # x = x.view(x.size()[0], -1)
        # x = x.view(-1,2*2*512)
        x = self.avgpool(x)
        x = x.reshape(input.shape[0], -1)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # output = self.softmax(self.fc3(x))
        
        output = self.fc3(x)
        # output = self.relu(output)
        # output = self.sigmoid(output)
        return output
    

class OSP(nn.Module):
    def __init__(self, args):
        super(OSP, self).__init__()
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        self.osp_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
                for _ in range(self.num_layers - 1)],
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, inputs):
        out = self.osp_linear(inputs)

        return out