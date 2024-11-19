import torch
import torchvision
import numpy as np
import SimpleITK as sitk
from torch import nn
import os
import math

# import sys
# sys.path.append('../')
# import utils.network_utils

import network_utils

from argparse import ArgumentParser
# from tensorboardX import SummaryWriter
from datetime import datetime as dt
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from networks import *

os.environ["CUDA_VISIBLE_DEVICES"] = '4'



def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--data_root_train',  default='/data/wrx/code/DVR-NeRF/dataset/mri/my3d_t/patient003/wrx4d_cropped.nii.gz')
    parser.add_argument('--data_root_test',  default='/data/wrx/code/DVR-NeRF/dataset/mri/my3d_t/patient003/wrx4d_cropped.nii.gz')
    parser.add_argument('--NumSlice',  default=256, type=int)
    parser.add_argument('--gpu',  help='GPU device id to use [cuda0]', default='0', type=str)
    parser.add_argument('--rand',  help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test',  help='Test neural networks', action='store_true')
    parser.add_argument('--train_batch_size', help='name of the net', default=1, type=int)
    parser.add_argument('--test_batch_size', help='name of the net', default=1, type=int)
    parser.add_argument('--plane', help='axial, coronal, and sagittal', default= 'coronal')# 修改层面需要同时修改数据集读入的维度及顺序，以及网络模型conv1的通道数
    parser.add_argument('--network', help='ResNet50, VGG19, and etc.', default= 'Linear')
    parser.add_argument('--num_worker',  default=4, type=int)
    parser.add_argument('--lr',  default=0.001, type=float)
    parser.add_argument('--betas',  default=(.9, .999), type=float)
    parser.add_argument('--lr_milestone', default=[50], type=int)
    parser.add_argument('--total_epoches', default=1000, type=int)
    parser.add_argument('--save_frequence', default=100, type=int)
    parser.add_argument('--weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out_dir', help='Set output path', default='./output/')
    parser.add_argument('--checkpoint_dir', default='./0612_osp_3/')
    parser.add_argument('--fig_dir', default='./osp_pattern_test_3/')
    parser.add_argument('--input_dim',  default=10000, type=int) 
    parser.add_argument('--output_dim',  default=100, type=int)
    parser.add_argument('--hidden_dim',  default=256, type=int)
    parser.add_argument('--num_layers',  default=8, type=int)
    args = parser.parse_args()
    return args

def normalize(data):
    if (data.max()-data.min()) == 0:
        return data
    else:
        return ((data-data.min())/(data.max()-data.min()))

class VSDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, args, data_root, transforms=None, mode='2d', plane='axial'):
        self.args = args
        self.data_root = data_root
        self.transforms = transforms
        self.mode = mode
        self.plane = plane
        if self.mode == '2d':
            self.volume = sitk.GetArrayFromImage(sitk.ReadImage(self.data_root))
            print(self.volume.shape)
            self.ShuffleIdx = np.random.permutation(np.array([i for i in range(self.volume.shape[1])], dtype=np.int))
        elif self.mode == '3d':
            self.VolumeList = os.listdir(self.data_root)
        elif self.mode == '4d':
            
            self.volumes_4d = normalize(sitk.GetArrayFromImage(sitk.ReadImage(self.data_root)))
            self.num_phases = self.volumes_4d.shape[-1]
    
    def __len__(self):
        if self.mode == '2d':
            return self.volume.shape[1]
        elif self.mode == '3d':
            return len(self.VolumeList)
        elif self.mode == '4d':
            return self.num_phases
        
    
    

    def __getitem__(self, idx):
        if self.mode == '2d':
            # volume = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, self.VolumeList[idx])))
            # self.volume = self.volume.transpose((1,0,2))
            VolumeShape = self.volume.shape  # N,H,W
            # Define the axial, sagittal, and coronal plane
            slice = self.volume[:,self.ShuffleIdx[idx],:]
            slice = np.expand_dims(slice, axis=0)
            
            return slice, self.ShuffleIdx[idx]/self.ShuffleIdx.shape[0]
            
        elif self.mode == '3d':
            OriginalVolume = normalize(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, self.VolumeList[idx])))) 
            # self.volume = self.volume.transpose((1,0,2))
            VolumeShape = OriginalVolume.shape  # N,H,W
            # Define the axial, sagittal, and coronal plane
            if self.plane == 'axial':
                volume = OriginalVolume[:, int(VolumeShape[1]/2)-int(self.args.NumSlice/2):int(VolumeShape[1]/2)+int(self.args.NumSlice/2), :]
                NewShape = volume.shape
                ShuffleIdx = np.random.permutation(np.array([i for i in range(NewShape[1])], dtype=np.int))
                # print(ShuffleIdx)
                ShuffleVolume = np.zeros(shape=NewShape)
                for idx in range(NewShape[1]):
                    ShuffleVolume[:,idx,:] = volume[:,ShuffleIdx[idx],:]
                
                if self.transforms:
                    ShuffleVolume = self.transforms(ShuffleVolume)

                ShuffleVolume = np.swapaxes(ShuffleVolume,0,1)
                ShuffleVolume = np.expand_dims(ShuffleVolume, 0)  # c, d, h, w
                
                return ShuffleVolume, ShuffleIdx/NewShape[1]
            elif self.plane == 'coronal':
                volume = OriginalVolume[:, :, int(VolumeShape[2]/2)-int(self.args.NumSlice/2):int(VolumeShape[2]/2)+int(self.args.NumSlice/2)]
                NewShape = volume.shape
                np.random.seed(1)
                ShuffleIdx = np.random.permutation(np.array([i for i in range(NewShape[2])], dtype=np.int))
                # print(ShuffleIdx)
                ShuffleVolume = np.zeros(shape=NewShape)
                for idx in range(NewShape[2]):
                    ShuffleVolume[:,:,idx] = volume[:,:,ShuffleIdx[idx]]
                
                if self.transforms:
                    ShuffleVolume = self.transforms(ShuffleVolume)

                ShuffleVolume = np.swapaxes(ShuffleVolume,0,2)
                ShuffleVolume = np.expand_dims(ShuffleVolume, 0)  # c, d, h, w
                
                return ShuffleVolume, normalize(ShuffleIdx)#/NewShape[2]
            elif self.plane == 'sagittal':
                volume = OriginalVolume[int(VolumeShape[0]/2)-int(self.args.NumSlice/2):int(VolumeShape[0]/2)+int(self.args.NumSlice/2), :, :]
                NewShape = volume.shape
                ShuffleIdx = np.random.permutation(np.array([i for i in range(NewShape[0])], dtype=np.int))
                # print(ShuffleIdx)
                ShuffleVolume = np.zeros(shape=NewShape)
                for idx in range(NewShape[0]):
                    ShuffleVolume[idx,:,:] = volume[ShuffleIdx[idx],:,:]
                
                if self.transforms:
                    ShuffleVolume = self.transforms(ShuffleVolume)

                # ShuffleVolume = normalize(ShuffleVolume)
                # ShuffleIdx = normalize(ShuffleIdx)

                # ShuffleVolumeAxial = np.swapaxes(ShuffleVolumeAxial,0,1)
                ShuffleVolume = np.expand_dims(ShuffleVolume, 0)  # c, d, h, w
                
                
                return ShuffleVolume, ShuffleIdx/NewShape[0]
        
        elif self.mode == '4d':
            sh = self.volumes_4d.shape
            volume = self.volumes_4d[...,idx]   # 60, 60, 60
            volume_a = volume
            volume_s = np.swapaxes(volume, 0, 1)
            volume_c = np.swapaxes(volume, 0, 2)

            # volume_flat = np.reshape(volume_a, (volume.shape[0], -1))   # 60, 3600
            # volume_flat = np.reshape(volume_s, (volume.shape[0], -1))   # 60, 3600
            volume_flat = np.reshape(volume_c, (volume.shape[0], -1))   # 60, 3600

            order = np.arange(0,volume.shape[0],1)  # 60
            ShuffleIdx = np.random.permutation(order)
            shufflevolume = np.zeros_like(volume_flat)  # 60, 3600
            for i in range(shufflevolume.shape[0]):
                shufflevolume[i] = volume_flat[ShuffleIdx[i]]
            
            oh_shuffleidx = np.eye(volume_flat.shape[0])[ShuffleIdx]
            return shufflevolume, oh_shuffleidx





def smooth_l1_loss(input, target, sigma, reduce=True, normalizer=1.0):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer


def train(args):

    # 数据增强
    # train_transforms = utils.data_transforms.Compose([
    #     utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
    #     utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
    #     utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
    #     utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
    #     utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    #     utils.data_transforms.RandomFlip(),
    #     utils.data_transforms.RandomPermuteRGB(),
    #     utils.data_transforms.ToTensor(),
    # ])

    #定义数据集
    TrainDataset = VSDataset(args, data_root=args.data_root_train, mode='4d', plane=args.plane)
    #定义DataLoader
    train_data_loader = torch.utils.data.DataLoader(dataset=TrainDataset,
                                                    batch_size=args.train_batch_size,
                                                    num_workers=args.num_worker,
                                                    pin_memory=True,
                                                    shuffle=False,
                                                    drop_last=True)
    #定义模型
    # 1. ResNet50
    # model = torchvision.models.resnet50(pretrained=True)
    # model.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    # model.fc = torch.nn.Linear(in_features=2048, out_features=1)

    # model = ResNet50(Bottleneck, args.NumSlice)
    # print(model)

    # 2. VGG19
    models = {'VGG19': VGG19(args),
              'ResNet50': ResNet50(Bottleneck, args),
              'Linear': OSP(args)}
    model = models[args.network].cuda()
    print(model)

    # model.apply(utils.network_utils.init_weights)

    #定义优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 betas=args.betas)
    #定义学习率迭代器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_milestone,
                                                        gamma=.5)
    #挂载GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    #定义损失函数
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    huber_loss = torch.nn.SmoothL1Loss()
    bce_loss = torch.nn.BCELoss()

    # TensorBoard Summary writer
    # output_dir = os.path.join(args.out_dir, '%s', dt.now().isoformat())
    # log_dir = output_dir % 'logs'
    # ckpt_dir = output_dir % 'checkpoints'
    # pred_dir = output_dir % 'prediction'
    # train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    # val_writer = SummaryWriter(os.path.join(log_dir, 'test'))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    mismatch = np.zeros(shape=(100,))

    os.makedirs(args.fig_dir, exist_ok=True)
    # 训练epoch
    for epoch in range(args.total_epoches):
        losses = network_utils.AverageMeter()
        model.train()

        n_batches = len(train_data_loader)

        for batch_idx, (slice, order) in enumerate(train_data_loader):
            slice = slice.type(torch.FloatTensor).cuda()
            print(slice.shape)
            order = order.type(torch.FloatTensor).cuda()

            PredOrder = model(slice)
            # print(PredOrder)
            # print(order)
            # import pdb
            # pdb.set_trace()
            
            # loss = l1_loss(PredOrder, order)
            # loss = l2_loss(order, PredOrder)
            loss = huber_loss(order, PredOrder)
            # loss = bce_loss(PredOrder, order)   # y*log(y')+(1-y)*log(1-y')


            model.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d]  Loss = %.4f'
                % (dt.now(), epoch + 1, args.total_epoches, batch_idx + 1, n_batches, loss.item()))
            
        
        # index = np.argsort(order.detach().cpu().numpy())
        index = np.argmax(order.detach().cpu().numpy(), axis=1)
        ori_index = np.argsort(index)
        pred_index = np.argmax(PredOrder.detach().cpu().numpy(), axis=1)
        if epoch % 10 == 0:
            print(index)
            print(pred_index)

        cur_mismatch = []
        for i in ori_index[0]:
            cur_mismatch.append(np.abs(index[0][i] - pred_index[0][i]))
        cur_mismatch = np.array(cur_mismatch)
        mismatch += cur_mismatch
        
        # for i in index:
        #     mismatch.append(np.abs(order.detach().cpu().numpy()[0][i]*256 - PredOrder.detach().cpu().numpy()[0][i]*256))
        # mismatch = np.array(mismatch).squeeze(0)
        # import pdb
        # pdb.set_trace()

        fig1 = plt.figure()
        x = list(np.arange(0,order.shape[1]))
        plt.plot(x, cur_mismatch, marker='*', color='r', label='MisMatch between pred and gt')
        plt.yticks(np.arange(0,100,10))
        # plt.plot(x, np.zeros(shape=len(x)), linestyle='--', color='black', label='match line')
        # plt.show()
        fig1.savefig(os.path.join(args.fig_dir, str(epoch) + '_mismatch.png'))

        lr_scheduler.step()
        
        if (epoch + 1) % args.save_frequence == 0:
            torch.save(model, os.path.join(args.checkpoint_dir, 'ckpt-epoch-' + str(epoch)+'.pth'))

            fig2 = plt.figure()
            x = list(np.arange(0,order.shape[1]))
            plt.plot(x, (mismatch/epoch), marker='*', color='r', label='MisMatch between pred and gt')
            average_mismatch = np.array(mismatch/(epoch+1)).astype(int)
            plt.yticks(np.arange(0, average_mismatch.max()))
            # plt.plot(x, np.zeros(shape=len(x)), linestyle='--', color='black', label='match line')
            # plt.show()
            fig2.savefig(os.path.join(args.fig_dir, str(epoch) + '_mismatch_average.png'))
    import pdb
    pdb.set_trace()


def test(args):
    TestDataset = VSDataset(args, data_root=args.data_root_test, mode='3d', plane=args.plane)
    TestDataLoader = torch.utils.data.DataLoader(dataset=TestDataset,
                                                    batch_size=args.test_batch_size,
                                                    num_workers=args.num_worker,
                                                    pin_memory=True,
                                                    shuffle=False,
                                                    drop_last=True)
    # model = torchvision.models.resnet50(pretrained=False)
    # model.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    # model.fc = torch.nn.Linear(in_features=2048, out_features=1)
    # model.cuda()
    # model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'ckpt-epoch-99.pth')).module.state_dict())

    # ResNet50
    # model = ResNet50(Bottleneck, args.NumSlice).cuda()
    # model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'ckpt-epoch-99.pth')).module.state_dict())
    # print(model)
    models = {'VGG19': VGG19(args),
              'ResNet50': ResNet50(Bottleneck, args)}
              
    model = models[args.network].cuda()
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'ckpt-epoch-169.pth')).module.state_dict())
    print(model)

    model.eval()
    os.makedirs(args.fig_dir, exist_ok=True)
    for sample_idx, (slice, order) in enumerate(TestDataLoader):
        with torch.no_grad():
            slice = slice.type(torch.FloatTensor).cuda()
            print(slice.shape)
            # order = order.type(torch.FloatTensor).cuda()

            PredOrder = model(slice)
            PredOrder = PredOrder.squeeze(1).detach().cpu().numpy()
            print(order)
            print(PredOrder)
            print(order.shape[0])
            PredOrder *= order.shape[1]
            PredOrder = PredOrder.astype(np.int)
            PredOrder = np.squeeze(PredOrder,0)

            order *= order.shape[1]
            order = order.detach().cpu().numpy().astype(np.int)
            order = np.squeeze(order, 0)

            # print(order)
            # print(PredOrder)

            fig, axs = plt.subplots(2,2)
            # import pdb
            # pdb.set_trace()
            x = list(np.arange(0,order.shape[0]))
            RR = [100,200]
            for i in range(2):
                axs[0, i].plot(x[0+i*128:128+i*128], order[0+i*128:128+i*128], marker='o', color='r', label='GT_Order')
                axs[0, i].plot(x[0+i*128:128+i*128], PredOrder[0+i*128:128+i*128], marker='*', color='b', label='Pred_Order')
                
                axs[0, i].legend()

                axs[1, i].plot(x[256+i*128:384+i*128], order[256+i*128:384+i*128], marker='o', color='r', label='GT_Order')
                axs[1, i].plot(x[256+i*128:384+i*128], PredOrder[256+i*128:384+i*128], marker='*', color='b', label='Pred_Order')
                axs[1, i].legend()
            # axs[0].plot(x[RR[0]:RR[1]], order[RR[0]:RR[1]], marker='o', color='r', label='GT_Order')
            # axs[1].plot(x[RR[0]:RR[1]], PredOrder[RR[0]:RR[1]], marker='*', color='b', label='Pred_Order')
            # axs[2].plot(x[RR[0]:RR[1]], order[RR[0]:RR[1]], marker='o', color='r', label='GT_Order')
            # axs[3].plot(x[RR[0]:RR[1]], PredOrder[RR[0]:RR[1]], marker='*', color='b', label='Pred_Order')
            # plt.show()
            fig.savefig(os.path.join(args.fig_dir, str(sample_idx) + '_compare.png'))

            mismatch = []
            index = np.argsort(order)
            for i in index:
                mismatch.append(np.abs(order[i] - PredOrder[i]))
            mismatch = np.array(mismatch)
            fig1 = plt.figure()
            plt.plot(x, mismatch, marker='*', color='r', label='MisMatch between pred and gt')
            plt.plot(x, np.zeros(shape=len(x)), linestyle='--', color='black', label='match line')
            # plt.show()
            fig1.savefig(os.path.join(args.fig_dir, str(sample_idx) + '_mismatch.png'))
            # import pdb
            # pdb.set_trace()
            print('Prediction: ', PredOrder)
            print('Ground Truth: ', order)

            



def optimal_scan_plane(volume, mode=0, view=0):
    # 提取最优扫描平面，共有3种模式选择
    # 0：直接选取最中间层作为最优扫描平面
    # 1：计算每层slice和完整volume的特征并映射到二维（或将二维slice特征concat到三维），选择与其最相似的层作为最优扫描平面
    # 2：target-specific 首先对重建目标进行分割，选择目标面积最大的层面（适用于显著目标，复杂目标不适用）
    
    vshape = volume.shape
    if mode == 0:
        mid_axial = int(vshape[0]/2)
        mid_coronal = int(vshape[1]/2)
        mid_sagittal = int(vshape[2]/2)
        osp_axial = volume[mid_axial,:,:]
        osp_coronal = volume[:,mid_coronal,:]
        osp_sagittal = volume[:,:,mid_sagittal]

    elif mode == 1:
        in_volume = torch.tensor(volume).unsqueeze(0).swapaxes(1,3).float().cuda()
        print(in_volume.shape)
        # import pdb
        # pdb.set_trace()

        res50v = torchvision.models.resnet50(pretrained=True).cuda()
        res50v.conv1 = nn.Conv2d(in_volume.shape[1], 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False).cuda()
        res50v.fc = nn.Linear(in_features=2048, out_features=1).cuda()
        res50v.eval()

        # vgg19 = torchvision.models.vgg19(pretrained=True)
        # vgg19.features[0] = nn.Conv2d(in_volume.shape[1], 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # import pdb
        # pdb.set_trace()
        
        # 计算volume特征
        with torch.no_grad():
            v_feature = res50v(in_volume)
            fv = v_feature.detach().cpu().numpy()
        # v_feature = vgg19.features(in_volume)
        print("Volume Feature: ", fv)
        
        #计算slice特征
        res50s = torchvision.models.resnet50(pretrained=True).cuda()
        res50s.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False).cuda()
        res50s.fc = nn.Linear(in_features=2048, out_features=1).cuda()
        res50s.eval()
        # vgg19.features[0] = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        similarity = dict()
        pick = 0
        
        for idx in range(in_volume.shape[1]):
            with torch.no_grad():
                # import pdb
                # pdb.set_trace()
                slice = in_volume[:,:,idx,:].unsqueeze(1)
                # print(slice.shape)
                s_feature = res50s(slice)
                fs = s_feature.detach().cpu().numpy()
                # s_feature = vgg19.features(slice)
                print("The {}-th Slice Feature: {}".format(idx, fs))
                # cos = torch.cosine_similarity(v_feature, s_feature)
                diff = math.pow((fv - fs),2)
                # import pdb
                # pdb.set_trace()
                similarity.update({idx:diff})

            

            # similarity.append(cos)
        sort_sim = sorted(similarity.items(), key=lambda x: x[1], reverse=False)
        print(sort_sim)
        

    # if view == 0:
    #     return osp_axial, osp_coronal, osp_sagittal
    # elif view == 1:
    #     return osp_axial
    # elif view == 2:
    #     return osp_coronal
    # elif view == 3:
    #     return osp_sagittal
    


if __name__ == '__main__':
    args = get_args_from_command_line()
    # import pdb
    # pdb.set_trace()
    # path = '/home/cj_group/Desktop/data/ocmr/4dmr/fs_0012_3T/phase_0.nii.gz'
    path = '/home/cj_group/Desktop/data/WHS/mr_train_images/'
    # volume = sitk.GetArrayFromImage(sitk.ReadImage(path))
    # print(volume.shape)
    
    
    # axial = normalize(volume[:,256,:])*255
    # sagittal = normalize(volume[:,:,256])*255
    # cv2.imwrite('axial.png', axial)
    # cv2.imwrite('sagittal.png', sagittal)
    train(args)
    # test(args)

    # def check_data(path):
    #     file_list = os.listdir(path)
    #     for file in file_list:
    #         data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, file)))
    #         print(file)
    #         print(data.shape)
    # check_data(path)
            