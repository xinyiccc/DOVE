import torch
import argparse
import os
#gpus = [0]
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import numpy as np
import mne
import math

import random

import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy.io import loadmat,savemat

import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

import torch.nn as nn
import torch.nn.functional as F
class MixedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super(MixedConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,  groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class VarPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(VarPool2d, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride )
        self.avgpool_sqr = nn.AvgPool2d(kernel_size, stride)

    def forward(self, input):
        # 平均池化
        mu = self.avgpool(input)
        # 均值平方池化
        mu_sqr = self.avgpool_sqr(input ** 2)
        # 计算方差
        var = mu_sqr - mu ** 2
        return var


# Convolution module
# use conv to capture local features, instead of postion embedding.
class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    strideFactor:每组方差的数据量
    step:窗口的移动步长
    '''
    def __init__(self, dim, strideFactor,step):
        super(LogVarLayer, self).__init__()
        self.dim = dim
        self.strideFactor=strideFactor
        self.step=step
        self.projection = nn.Sequential(
            nn.dropout(0.5),  # transpose, conv could enhance fiting ability slightly
            nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),  # 将 （k,1,m）->(m,k)
        )

    def forward(self, x):
         x = x.reshape([x.shape[0],x.shape[1], self.strideFactor, int(x.shape[3]/self.strideFactor)])
         x = torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6)) #最里层求方差，第二层取上下界，第三层取对数
         x = self.projection(x)
         return x
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):

        super().__init__()

        self.shallownet = nn.Sequential(
            
            #MixedConv2d(1,40,kernel_size=(1,25)),
            nn.Conv2d(1, 40, (1, 25), (1, 1)),#时间卷积
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            VarPool2d((1,75),(1,15)),
            #nn.AvgPool2d((1, 75), (1, 15)), 
            nn.Dropout(0.5),
                              # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),  #将 （k,1,m）->(m,k)
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module): #masked positional encoding
    def __init__(self, emb_size, num_heads, dropout, mask_value=-1e7):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.mask_value = mask_value
        max_sequence_length=128
        self.max_sequence_length = max_sequence_length

        # Positional Encoding
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pe = torch.zeros((max_sequence_length, emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # Linear layers
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = x + self.pe[:, :x.size(1), :]
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        mask = torch.tril(torch.ones((x.size(0),10,61,4)))
        mask = mask.to(device)
        # Apply the mask
        if mask is not None:
            keys = keys * mask
            values = values * mask


        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out



class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=5, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )
class ExP():



    def __init__(self, nsub):
        super(ExP, self).__init__()
        #self.batch_size0 = 72
        self.batch_size = 144
        self.n_epochs = 2000
        #self.n_epochs_sec = 1500
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50) #？
        self.nSub = nsub

        self.start_epoch = 0
        self.root = '/data/private/DOVE/bci-iv-2a/'
        self.log_write = open("/data/private/DOVE/results/log_subject%d.txt" % self.nSub, "w")


        self.Tensor = torch.FloatTensor()
        self.LongTensor = torch.LongTensor()

        self.criterion_l1 = torch.nn.L1Loss().to(device)
        self.criterion_l2 = torch.nn.MSELoss().to(device)
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(device)

        self.model = Conformer().to(device) #Conformer()这个是模型
        self.train_ratio = 0.8



    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).to(device)
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).to(device)
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_data(self,datapath,labelpath,epochWindow=[0, 4], chans=list(range(22))):
        eventCode = ['768']  # start of the trial at t=0
        fs = 250
        offset = 2

        raw_gdf = mne.io.read_raw_gdf(datapath, stim_channel="auto")
        raw_gdf.load_data()
        gdf_event_labels = mne.events_from_annotations(raw_gdf)[1]
        eventCode = [gdf_event_labels[x] for x in eventCode]

        gdf_events = mne.events_from_annotations(raw_gdf)[0][:, [0, 2]].tolist()
        eeg = raw_gdf.get_data()
        # drop channels
        if chans is not None:
            eeg = eeg[chans, :]

        # Epoch the data
        events = [event for event in gdf_events if event[1] in eventCode]
        y = np.array([i[1] for i in events])
        epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs
        x = np.stack([eeg[:, epochInterval + event[0]] for event in events], axis=2)

        # Multiply the data with 1e6
        x = x * 1e6

        # have a check to ensure that all the 288 EEG trials are extracted.
        assert x.shape[
                   -1] == 288, "Could not extracted all the 288 trials from GDF file: {}. Manually check what is the reason for this".format(
            datapath)

        # Load the labels
        y = loadmat(labelpath)["classlabel"].squeeze()

        y = y-1
        data = {'data': x, 'label': y}
        return  data

    def get_source_data(self):

        # train data
        self.total_data = self.get_data(self.root+'A0%dT.gdf' % self.nSub,self.root+'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 0, 1))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label

        shuffle_num = np.random.permutation(len(self.allData))

        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # test data
        self.test_tmp =  self.get_data(self.root+'A0%dE.gdf' % self.nSub,self.root+'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        self.test_data = np.transpose(self.test_data, (2, 0, 1))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label


        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel


    def train(self):
        train_data, train_label, test_data, test_label = self.get_source_data()
        train_data = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label)

        dataset = torch.utils.data.TensorDataset(train_data, train_label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = test_data. to(device).float()
        test_label = test_label.to(device).long()

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = img.to(device).float()
                label = label.to(device).long()

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))


                tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # out_epoch = time.time()


            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)


                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred


        torch.save(self.model.module.state_dict(), 'model.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred
        writer.close()


def main():
    best = 0
    aver = 0
    result_write = open("/data/private/DOVE/results/sub_result.txt", "w")

    for i in range(9):
        starttime = datetime.datetime.now()


        seed_n = np.random.randint(2021)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)


        print('Subject %d' % (i+1))
        exp = ExP(i + 1)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / 9
    aver = aver / 9

    print('the best average is:',best)

if __name__ == "__main__":
    '''
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))'''

    start_time = time.time()
    main()

    end_time = time.time()
    run_time = end_time - start_time
   
    hours = int(run_time / 3600)
    minutes = int((run_time % 3600) / 60)
    seconds = int(run_time % 60)

    # print the run time
    print("Program run time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))