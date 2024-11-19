import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import SimpleITK as sitk

from load_mri_abalation import *
from run_nerf_helpers_ablation import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] '
                           '- %(levelname)s: %(message)s',level=logging.DEBUG)
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_device = torch.device("cpu")   # 用cpu进行数据处理
np.random.seed(0)
# torch.random.seed(0)
DEBUG = False

# global time_embedding
# global cnt_embedding 
# global time_run_nerf 
# global cnt_run_nerf 
# global time_run_network 
# global cnt_run_network 
# global time_batch_ray 
# global cnt_batch_ray 


def batchify(fn, chunk):
    '''
    当输入数据过多，模型计算内存不够同步处理时，将输入分成n个chunk大小的batch，分别送入网络进行计算，然后将结果cat起来
    '''
    if chunk is None:
        return fn
    def ret(inputs_embed):  
        volume = []
        for i in range(0, inputs_embed.shape[0], chunk):
            res = fn(inputs_embed[i:i+chunk])
            volume.append(res)
        return torch.cat(volume, 0)

        # return torch.cat([fn(inputs_embed_0[i:i+chunk], inputs_pts_0[i:i+chunk], inputs_embed[i:i+chunk], inputs_pts[i:i+chunk]) for i in range(0, inputs_embed.shape[0], chunk)], 0)
    return ret

def run_network(inputs, time_info, fn, embed_fn, embedtime_fn, netchunk=1024*64):
    '''
    1. 用于调整输入维度，使其可以传入网络中进行计算
    2. 进行批处理
    3. 调用网络进行计算
    inputs: N_rays, 8
    time_info: N_rays,4
    '''
    # global time_embedding, time_run_nerf, cnt_embedding, cnt_run_nerf
    # torch.cuda.synchronize()
    # t = time.time()
    inputs_flat = torch.reshape(inputs[...,:4], [-1, 4]) # 1 024*64, 1
    # inputs_flat = torch.reshape(inputs, [-1, 4]) # 1 024*64, 1
    # embedded_spatial = embed_fn(inputs_flat)    # 1024*64, 21

    if time_info is not None:
        inputs_time = time_info[:,None].expand(inputs[...,4:].shape)
        # inputs_time = time_info[:,None].expand(inputs.shape)
        inputs_time_flat = torch.reshape(inputs_time, [-1, inputs_time.shape[-1]])
        # embedded_temporal = embedtime_fn(inputs_time_flat)  # 1024*64, 64
        # embedded = torch.cat([embedded_spatial, embedded_temporal], -1)     # 1024*64, 85

    embedded = torch.cat([inputs_flat, inputs_time_flat], -1)
    # torch.cuda.synchronize()
    # logging.info('Time Consumption for embedding: {}'.format(time.time() - t))
    # time_embedding = time_embedding + (time.time() - t)
    # cnt_embedding += 1


    # outputs_flat = batchify(fn, netchunk)(embedded) # 1024*64, 1
    # start = time.time()
    # print('start: ', start)

    # torch.cuda.synchronize()
    # t = time.time()
    outputs_flat = fn(embedded)
    # torch.cuda.synchronize()
    # logging.info('Time Consumption for run nerf model: {}'.format(time.time() - t))
    # time_run_nerf = time_run_nerf + (time.time() - t)
    # cnt_run_nerf += 1
    # print('end: ', time.time()-start)
    # for name, module in fn.named_modules():
    #     print(name)
    #     print(module)
    #     x = module(embedded)
    # outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])   # 1024, 64, 1
    return outputs_flat

def batchify_rays(pts_flat, chunk=1, **kwargs):
    # global time_batch_ray, cnt_batch_ray
    all_ret = []
    for i in range(0, pts_flat.shape[0], chunk):
        # t = time.time()
        # torch.cuda.synchronize()
        # t = time.time()
        ret = render_rays(pts_flat[i:i+chunk], **kwargs)
        # torch.cuda.synchronize()
        # logging.info('Total Time Consumption for batch ray: {}'.format(time.time() - t))
        # logging.info('Time Consumption for batch rays: {}'.format(time.time() - t))
        # time_batch_ray = time_batch_ray + (time.time() - t)
        # cnt_batch_ray += 1
        all_ret.append(ret)
        # for k in ret:
        #     if k not in all_ret:
        #         all_ret[k] = []
        #     all_ret[k].append(ret[k])
    # all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    all_ret = torch.cat(all_ret, 0)
    return all_ret  

def render(D, H, W, s_t, chunk=None, neigh=10, test=False, **kwargs):
    '''
    s_t: 2, N_rand, 1
    '''
    sh = (D, H, W)

    pts_slice, pts_time = s_t
    neigh_i = neigh * torch.ones_like(pts_time)
    pts_slice = torch.reshape(pts_slice, [-1,4]).float()
    pts_time = torch.reshape(pts_time, [-1,4]).float()
    pts_neigh = torch.reshape(neigh_i, [-1,4]).float()
    pts_flat = torch.cat([pts_slice, pts_neigh[...,0:1], pts_time], -1)  # N_rand, 3
    
    # torch.cuda.synchronize()
    # t = time.time()
    all_ret = batchify_rays(pts_flat, chunk, **kwargs)
    # torch.cuda.synchronize()
    # logging.info('Total Time Consumption for all batch rays: {}'.format(time.time() - t))
    
    if test == True:
        all_ret = torch.reshape(all_ret, sh)
        # for k in all_ret:
        #     all_ret[k] = torch.reshape(all_ret[k], sh)
    # k_extract = ['volume_img', 'volume_k']
    # ret_list = [all_ret[k] for k in k_extract]
    # ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return all_ret
    
    

def render_path(D, H, W, input_slices, input_times, chunk, render_kwargs, gt=None, savedir=None):
    '''
    input_slices: N,H,W
    input_times: N,1
    '''
    count_timestep = input_times.shape[0]
    # volumes_k = []
    volumes_img = []
    t = time.time()

    for i, (i_slice, i_time) in enumerate(tqdm(zip(input_slices, input_times))):
        # slice: H,W / time: 1
        coords_slice, coords_time = get_coords(D, H, W, i_slice, i_time)
        pts_slice = torch.reshape(coords_slice, [-1,1])
        pts_time = torch.reshape(coords_time, [-1,1])
        test_pts = torch.stack([pts_slice, pts_time], 0)

        # logging.info('Time Consumption: {}'.format(time.time() - t))
        torch.cuda.synchronize()
        t = time.time()
        volume_img = render(D, H, W, s_t=test_pts, chunk=chunk, test=True, **render_kwargs)
        torch.cuda.synchronize()
        logging.info('Total Time Consumption for a volume: {}'.format(time.time() - t))

        volumes_img.append(volume_img.detach().cpu().numpy())

    # volumes_k = np.stack(volumes_k, 0)
    volumes_img = np.stack(volumes_img, 0)
    return volumes_img

def create_nerf(D,H,W,args):
    '''
    实例化nerf模型
    原版nerf用的是MLP,但是用在这里可能会不够
    '''
    # 对输入进行positional embedding，使得模型可以处理高频信息
    # 具体方法是将输入进行傅里叶变换
    embed_fn, input_ch_s = get_embedder(args.multires, 4, args.i_embed) # , embedding方式
    embedtime_fn, input_ch_time = get_embedder(args.multires, 4, args.i_embed)
    # 我们这里需要对time进行embedding处理，旨在为模型提供更丰富的用于判断当前时刻的信息
    # embedtime_fn = Time_Embedder(input_dim=4, embed_dim=32)
    # input_ch_time = embedtime_fn.get_outdim()


    input_ch = input_ch_s + input_ch_time
    output_ch = 1 # pixel value in MRI/CT
    skips = [5]
    # 定义主要模型
    # 单纯定义基层MLP的话可能性能不足以支持重建出完整3D Volume，但是先试试，不行再加
    model = NeRF(D=args.netdepth, W=[128, 256, 256, 512, 256, 256, 256, 256],
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                spatial_dim=4, temporal_dim=4, hidden_dim=256, num_layers=8, region=args.N_samples).to(device)
    

    model = nn.DataParallel(model)
    model = model.cuda()
    # torch.distributed.ini t_process_group(backend="nccl")
    # model = nn.parallel.DistributedDataParallel(model)
    grad_vars = list(model.parameters())
    model_fine = None
    print(model)
    # if args.N_importance > 0:
    #     model_fine = NeRF(D=args.netdepth, W=args.netwidth,
    #             input_ch=input_ch, output_ch=output_ch, skips=skips,
    #             spatial_dim=1, temporal_dim=1, hidden_dim=128, num_layers=8).to(device)
    #     grad_vars += list(model_fine.parameters())
    
    network_query_fn = lambda inputs, time_info, network_fn : run_network(inputs, time_info, 
                                                                            network_fn, 
                                                                            embed_fn=embed_fn,
                                                                            embedtime_fn=embedtime_fn,
                                                                            netchunk=args.netchunk)
    
    # 定义优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(.9, .999), weight_decay=1e-4)

    start = 0
    basedir = args.basedir
    expname = args.expname
    
    # load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logging.info('Found ckpts: {}'.format(ckpts))
    if len(ckpts) >= 1 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logging.info('Reloading from: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
         
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,    # maybe a useful hyperparameter
        'network_fn' : model,
        # 'use_viewdirs' : args.use_viewdirs,
        # 'white_bkgd' : args.white_bkgd,
        # 'raw_noise_std' : args.raw_noise_std,
    }
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def raw2outputs(raw):
    '''
    这里传入的参数raw是模型的直接输出，在VR-NeRF中是K空间域的值，我们需要通过傅里叶变换将其变换到图像域，尺寸维度保持不变
    raw: 1024, 64, 1    1024, D*H*W, 1  -> sum(, 1) -> 1024, 1
    这里64是想模拟4*4*4空间内的点，即局部空间的点贡献于某一点的灰度值
    '''

    volume_img = raw   # 1024, 1   k-space
    volume_k = torch.abs(torch.fft.ifftn(volume_img)) # 1024, 1 image
    return volume_img, volume_k

def render_rays(pts_batch, # N_chunk, 3
                network_fn, 
                network_query_fn, 
                retraw=False,
                N_samples=0,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    '''
    进行volumetric reconstruction的实际函数
    '''
    # global time_run_network, cnt_run_network
    N_pts = pts_batch.shape[0] # num of chunk
    pts_slice, pts_neigh, pts_time = pts_batch[:,0:4], pts_batch[:,4:5], pts_batch[:,5:]
    pts_slice = torch.reshape(pts_slice, [-1,4]).float()
    pts_time = torch.reshape(pts_time, [-1,4]).float()
    pts_neigh = torch.reshape(pts_neigh, [-1,1]).float()
    region = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        region = pts_neigh * region
    
    #对region进行随机采样，
    pts = torch.cat([pts_slice[...,None,:] * region[...,:,None], pts_time[...,None,:] * region[...,:,None]], -1) 
    # pts = pts_slice[...,None,:] * region[...,:,None]

    # torch.cuda.synchronize()
    # t = time.time()
    raw = network_query_fn(pts, pts_time, network_fn)   # B,1
    # torch.cuda.synchronize()
    # logging.info('Time Consumption for run network: {}'.format(time.time() - t))
    # time_run_network = time_run_network + (time.time() - t)
    # cnt_run_network += 1
    # volume_img, volume_k = raw2outputs(raw)
    
    # ret = {'volume_img': volume_img, 'volume_k': volume_k}
    # if retraw:
    #     ret['raw'] = raw
    # for k in ret:
    #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
    #         print(f'! [Numerical Error] {k} contains nan or inf.')
    
    return raw

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, default='1_ablation_wope',
                        help='experiment name')
    parser.add_argument("--patient_id", type=int, default=1,
                        help='patient id')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='../dataset/mri/my3d_t', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=10, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=1024*8, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=1e-3, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=20000, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*100, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*3*3*3, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", #default=True, 
                        action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--use_batching", type=bool, default=True,
                        help='whether to batchify the input coordinate when inputting into network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=2*2*2, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_neigh", type=int, default=3, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", #default=True,
                            action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='mri', 
                        help='options: mri / ct')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=10, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=2500, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--local-rank", default=-1, type=int)

    return parser

def train():
    parser = config_parser()
    args = parser.parse_args()  # 各种参数

    # 准备数据，nerf的逻辑是把所有训练数据先读到内存里，但是要看数据和内存大小是否支持
    '''
    1. 目前我们只需要处理MRI数据。数据格式为：
        输入：一张或多张2D MRI断层图像，并指定给定2D图像所属运动时刻t (H,W)+embedding(t)
        输出：当前时刻t下目标完整3D Volume (D,H,W)
    2. 后续可以设置多种不同MRI序列的数据处理方法，具体差异由扫描方式决定,如gre，trufi，golden angle等
    '''
    if args.dataset_type == 'mri':
        # volumes_img, volumes_k, slices_img, slices_k, times = load_mri_data(args.datadir)    # N,D,H,W   N,H,W   N,1

        trainset, testset = load_mri_data(args.datadir, args.patient_id)
        # trainset, testset = load_mri_data_singleview(args.datadir)
        volumes_img, volumes_k, slices_imgs, slices_k, times = trainset
        testvolumes_img, testvolumes_k, testslices_imgs, testslices_k, testtimes = testset

        logging.info('MRI Data loaded with shape: {}, {}, {}'.format(volumes_k.shape, slices_k.shape, times.shape))

    # 目标图像基本尺寸维度信息
    total_N, D, H, W = volumes_k.shape[0], volumes_k.shape[1], volumes_k.shape[2], volumes_k.shape[3]

    # 保存当次训练config信息
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    log_file = logging.FileHandler(filename=os.path.join(basedir, expname, 'logs.txt'), encoding='utf-8', mode='a+')
    logger.addHandler(log_file)

    # 定义nerf模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(D,H,W,args)
    gloabl_step = start

    neighbor_dict = {
        'neigh': args.N_neigh,
    }
    render_kwargs_train.update(neighbor_dict)
    render_kwargs_test.update(neighbor_dict)

    input_volumes_k = torch.tensor(volumes_k, dtype=torch.complex128).to(device)
    input_volumes_img = torch.tensor(volumes_img, dtype=torch.float64).to(device)
    input_slices_imgs = torch.tensor(slices_imgs, dtype=torch.float64).to(device)
    input_slices_k = torch.tensor(slices_k, dtype=torch.complex128).to(device)
    input_times = torch.tensor(times).to(device)

    input_test_volumes_k = torch.tensor(testvolumes_k, dtype=torch.complex128).to(device)
    input_test_volumes_img = torch.tensor(testvolumes_img, dtype=torch.float64).to(device)
    input_test_slices_imgs = torch.tensor(testslices_imgs, dtype=torch.float64).to(device)
    input_test_slices_k = torch.tensor(testslices_k, dtype=torch.complex128).to(device)
    input_test_times = torch.tensor(testtimes).to(device)

    # only perform reconstruction with trained model
    if args.render_only:
        logging.info('Render Only!')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('Input Slice Shape', input_slices_imgs[...,0].shape)

            # new_input_test_slices_imgs = torch.tile(input_test_slices_imgs[0], (input_test_slices_imgs.shape[0], 1, 1, 1))
            new_input_test_slices_imgs = torch.zeros_like(input_test_slices_imgs)
            recon_volumes_img = render_path(D, H, W, input_test_slices_imgs, input_test_times, args.chunk, render_kwargs_test)
            
            # print('Average time for embedding: {}, run nerf: {}, run network: {}, batch ray: {}'.format((time_embedding/cnt_embedding), (time_run_nerf/cnt_run_nerf),(time_run_network/cnt_run_network),(time_batch_ray/cnt_batch_ray)))

            mae, psnr, ssim = test_metrics(recon_volumes_img, input_test_volumes_img.cpu().numpy())
            print('MAE: {}, PSNR: {}, SSIM: {}'.format(mae, psnr, ssim))
            recon_volumes_img_out = np.swapaxes(recon_volumes_img, 0, 3)
            sitk.WriteImage(sitk.GetImageFromArray(recon_volumes_img_out), os.path.join(testsavedir, 'test_4d.nii.gz'))
            print('Finish Reconstruction', testsavedir)
            # for i in range(volumes_img.shape[0]):
            #     sitk.WriteImage(sitk.GetImageFromArray(volumes_img[i]), os.path.join(testsavedir, str(i).zfill(3)+'.nii.gz'))
            # imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(show2ds), fps=30, quality=8)
            return
    
    N_iters = 100000
    logging.info('Begin')
    # start = start + 1
    i_batch = 0
    sw_loss = SlicedWassersteinLoss()
    l1char_loss = L1_Charbonnier_loss()
    for i in range(start, N_iters):
        # data_loader   B=1
        # input at time t_0
        i_volume_k_0 = input_volumes_k[i_batch]   # D, H, W
        i_volume_img_0 = input_volumes_img[i_batch]   # D, H, W
        i_slice_k_0 = input_slices_k[i_batch]     # H, W
        i_slice_img_0 = input_slices_imgs[i_batch]     # H, W
        i_time_0 = input_times[i_batch]       # 1

        di = np.floor(np.random.rand(1)[0] * (total_N-i_batch-1)).astype(np.int_)

        # input at time t_i
        i_volume_k = input_volumes_k[i_batch+di]   # D, H, W
        i_volume_img = input_volumes_img[i_batch+di]   # D, H, W
        i_slice_k = input_slices_k[i_batch+di]     # H, W
        i_slice_img = input_slices_imgs[i_batch+di]     # H, W
        i_time = input_times[i_batch+di]       # 1
        i_batch += 1
        if i_batch >= total_N-2:
            i_batch = 0
        
        coords = torch.stack(torch.meshgrid(torch.linspace(0, D-1, D), torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (D, H, W, 3)
        coords_flat = torch.reshape(coords, [-1, 3])    # D*H*W, 3
        num_epoch = int(np.floor((D*H*W)/args.N_rand))   
        # random sample points
        sample_p = torch.flatten(torch.ones_like(i_volume_img))
        diff = torch.abs((i_volume_img - i_volume_img_0).flatten())
        sample_p[diff>0.01] = 10
        sample_p = torch.softmax(torch.tensor(sample_p),-1)
        for epoch in range(num_epoch):
            select_inds = np.random.choice(coords_flat.shape[0], size=[args.N_rand], replace=False, p=sample_p.cpu().numpy()) # N_rand, 
            sample_p[select_inds] = -1
            sample_p = torch.softmax(torch.tensor(sample_p),-1)
            # select_inds = np.random.choice(coords_flat.shape[0], size=[args.N_rand], replace=False) # N_rand, 
            select_coords = coords_flat[select_inds].long() # N_rand, 3

            # time t_0
            coords_slice_0, coords_time_0 = get_coords(D, H, W, i_slice_img_0, i_time_0)
            # coords_slice_0, coords_time_0 = get_coords_singleview(D, H, W, i_slice_img_0, i_time_0)
            pts_slice_0 = coords_slice_0[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]] # N_rand, 1 只有1维数据太少了，不足以反应三维coordinate
            pts_time_0 = coords_time_0[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]   # N_rand, 1
            batch_pts_0 = torch.stack([pts_slice_0, pts_time_0], 0) # 2, N_rand, 1
            target_s_k_0 = i_volume_k_0[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]
            target_s_img_0 = i_volume_img_0[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]
            # time t_i
            coords_slice, coords_time = get_coords(D, H, W, i_slice_img, i_time)
            # coords_slice, coords_time = get_coords_singleview(D, H, W, i_slice_img, i_time)            
            pts_slice = coords_slice[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]] # N_rand, 1 只有1维数据太少了，不足以反应三维coordinate
            pts_time = coords_time[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]   # N_rand, 1
            batch_pts = torch.stack([pts_slice, pts_time], 0) # 2, N_rand, 1
            target_s_k = i_volume_k[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]
            target_s_img = i_volume_img[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]


            volume_img_0 = render(D, H, W, batch_pts_0, chunk=args.chunk, **render_kwargs_train)
            volume_img= render(D, H, W, batch_pts, chunk=args.chunk, **render_kwargs_train)

            optimizer.zero_grad()
            # img_loss = img2mse(volume_img, target_s_img) + sw_loss(volume_img, target_s_img) + PerceptualLoss(volume_img, target_s_img)
            # 根据本次采样与重建结果，找到hard samples，即重建效果差的采样点，下次采样时在这些样本上多进行采样
            attention = torch.abs(target_s_img - target_s_img_0)
            mask = torch.ones_like(target_s_img)
            mask[attention>0.01] = 10
            img_loss_0 = weighted_mse_loss(torch.squeeze(volume_img_0), target_s_img_0, mask) + 10 *sw_loss(torch.squeeze(volume_img_0), target_s_img_0) #+ ssim_loss(torch.squeeze(volume_img_0), target_s_img_0) 
            img_loss = weighted_mse_loss(torch.squeeze(volume_img), target_s_img, mask) + 10 *sw_loss(torch.squeeze(volume_img), target_s_img) #+ ssim_loss(torch.squeeze(volume_img), target_s_img)
            temporal_loss = l1char_loss((volume_img - volume_img_0), (target_s_img - target_s_img_0)) 
            # img_loss_0 = img2mse(torch.squeeze(volume_img_0), target_s_img) + 10 *sw_loss(torch.squeeze(volume_img_0), target_s_img) #+ 0.1*k2mse(volume_img, target_s_img)
            # img_loss = img2mse(torch.squeeze(volume_img), target_s_img) + 10 *sw_loss(torch.squeeze(volume_img), target_s_img)
            loss = img_loss_0 + img_loss + temporal_loss
            psnr = mse2psnr(img2mse(torch.squeeze(volume_img), target_s_img)) #+ mse2psnr(k_loss)
            loss.backward(retain_graph=True)
            optimizer.step()

        
        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (gloabl_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if (i+1)%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': gloabl_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
            }, path)
            logging.info('Saved checkpoints at {}'.format(path))
        
        # if i%args.i_video==0 and i > 0:
        #     with torch.no_grad():
        #         volumes = render_path(input_slices, input_times, render_kwargs_test)
        #     # logging.info('Done, Saving', volumes.shape, show2ds.shape)
        #     show2ds = volumes[:,int(D/2),:,:]
        #     moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}'.format(i))
        #     imageio.mimwrite(moviebase + '2dmovie.mp4', to8b(show2ds), fps=30, quality=8)
        
        if i%args.i_print==0:
            t = time.gmtime()
            currenttime = time.strftime("%Y-%m-%d %H:%M:%S",t)
            # tqdm.write(f"[TRAIN] Duration:{currenttime} Iter: {i} Loss: {loss.item()} PSNR: {psnr.item()}  \
            #         Pred: {volume_img.detach().cpu().numpy().max(), volume_img.detach().cpu().numpy().mean()} GT: {target_s_img.mean()}")
            logger.info(f"[TRAIN] Duration:{currenttime} Iter: {i} Loss: {loss.item()} PSNR: {psnr.item()} Pred: {volume_img.detach().cpu().numpy().max(), volume_img.detach().cpu().numpy().mean()} GT: {target_s_img.mean()}")

        gloabl_step += 1

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()        