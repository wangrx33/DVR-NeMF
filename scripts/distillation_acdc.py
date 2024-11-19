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

from load_mri_distillation import *
from run_nerf_helpers_distillation import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "11"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_device = torch.device("cpu")   # 用cpu进行数据处理
np.random.seed(0)
DEBUG = False

import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] '
                           '- %(levelname)s: %(message)s',level=logging.DEBUG)
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)

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
        coords_slice, coords_time = get_coords_singleview(D, H, W, i_slice, i_time)
        pts_slice = torch.reshape(coords_slice, [-1,1])
        pts_time = torch.reshape(coords_time, [-1,1])
        test_pts = torch.stack([pts_slice, pts_time], 0)

        torch.cuda.synchronize()
        t = time.time()
        sfeat, tfeat, feature, volume_img = render(D, H, W, s_t=test_pts, chunk=chunk, test=False, render_only=True, **render_kwargs)
        torch.cuda.synchronize()
        logging.info('Total Time Consumption for a volume: {}'.format(time.time() - t))

        # volume_img = render(D, H, W, s_t=test_pts, chunk=chunk, test=False, render_only=True, **render_kwargs)
        # volume_img_real = torch.abs(torch.fft.ifftn(volume_k))
        # volumes_k.append(feature.cpu().numpy())
        volumes_img.append(volume_img.cpu().numpy())

    # volumes_k = np.stack(volumes_k, 0)
    volumes_img = np.stack(volumes_img, 0)
    return volumes_img

def render_only(D, H, W, s_t, chunk, neigh=10, test=True, **kwargs):
    with torch.no_grad():
        x, t, feature, img = render(D, H, W, s_t, chunk, test=True, **kwargs)
    
    return x, t, feature, img

def run_network(inputs, time_info, fn, embed_fn, embedtime_fn, spatial_dim, temporal_dim, test=False):
    inputs_flat = torch.reshape(inputs[...,:4], [-1, 4]) # 1 024*64, 1
    embedded_spatial = embed_fn(inputs_flat)    # 1024*64, 21

    if time_info is not None:
        inputs_time = time_info[:,None].expand(inputs[...,4:].shape)
        # inputs_time = time_info[:,None].expand(inputs.shape)
        inputs_time_flat = torch.reshape(inputs_time, [-1, inputs_time.shape[-1]])
        embedded_temporal = embedtime_fn(inputs_time_flat)  # 1024*64, 64
        embedded = torch.cat([embedded_spatial, embedded_temporal], -1)     # 1024*64, 85
    
    if test:
        spatial_input = embedded[:, :spatial_dim]
        temporal_input = embedded[:, temporal_dim:]
        x = spatial_input
        for i, l in enumerate(fn.module.spatial_encoder):
            x = fn.module.spatial_encoder[i](x)
            x = F.relu(x)
            if i in fn.module.skips:
                x = torch.cat([spatial_input, x], -1)

        # temporal_encoding = self.temporal_encoder(temporal_input)
        t = temporal_input
        for i, l in enumerate(fn.module.temporal_encoder):
            t = fn.module.temporal_encoder[i](t)
            t = F.relu(t)
            if i in fn.module.skips:
                t = torch.cat([temporal_input, t], -1)


        # encoding = torch.cat([x, t], dim=1)
        encoding = torch.add(x, t)

        radiance_field = fn.module.decoder(encoding) # B*27, 1
        render = radiance_field.view(-1, 8)   # B, 27
        out = fn.module.output(render)
        return x, t, render, out

    else:
        x, t, feature, output = fn(embedded)
        return x, t, feature, output
        # output = fn(embedded)
        # return output
        
    
    

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
                test,
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
                pytest=False,
                ):
    '''
    进行volumetric reconstruction的实际函数
    '''
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

    x, t, feat, img = network_query_fn(pts, pts_time, network_fn)   # B,1
    # img = network_query_fn(pts, pts_time, network_fn)   # B,1
    # volume_img, volume_k = raw2outputs(raw)
    
    ret = {'spatial': x, 'temporal': t,  'feature': feat, 'image': img}
    # ret = {'spatial': x, 'temporal': t,  'feature': feat, 'image': img}
    # if retraw:
    #     ret['out'] = img
    # for k in ret:
    #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
    #         print(f'! [Numerical Error] {k} contains nan or inf.')
    
    return ret

def batchify_rays(pts_flat, chunk=1, test=False, render_only=False, **kwargs):
    # if render_only:
    #     all_ret = {}
    #     ret = render_rays(pts_flat, test, **kwargs)
    #     for k in ret:
    #         if k not in all_ret:
    #             all_ret[k] = []
    #         all_ret[k].append(ret[k])
    #     all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    # else:
    all_ret = {}
    for i in range(0, pts_flat.shape[0], chunk):
        ret = render_rays(pts_flat[i:i+chunk], test, **kwargs)
        # all_ret.append(ret)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    # all_ret = torch.cat(all_ret, 0)
    return all_ret  


def render(D, H, W, s_t, chunk=None, neigh=10, test=False, render_only=False, **kwargs):
    '''
    s_t: 2, N_rand, 1
    '''
    sh = (D, H, W)
    neigh=10
    # if test:
    #     neigh=10
    # else:
    #     neigh=1

    pts_slice, pts_time = s_t
    neigh_i = neigh * torch.ones_like(pts_time)
    pts_slice = torch.reshape(pts_slice, [-1,4]).float()
    pts_time = torch.reshape(pts_time, [-1,4]).float()
    pts_neigh = torch.reshape(neigh_i, [-1,4]).float()
    pts_flat = torch.cat([pts_slice, pts_neigh[...,0:1], pts_time], -1)  # N_rand, 3

    all_ret = batchify_rays(pts_flat, chunk, test, render_only, **kwargs)

    if render_only == True:
        # all_ret = torch.reshape(all_ret, sh)
        for k in all_ret:
            if k == 'image':
                all_ret[k] = torch.reshape(all_ret[k], sh)
    k_extract = ['spatial', 'temporal', 'feature', 'image']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list
    

def create_teacher_student_nerf(args):

    # Teacher model setting
    embed_fn, input_ch_s = get_embedder(args.multires, 4, args.i_embed)
    embedtime_fn, input_ch_time = get_embedder(args.multires, 4, args.i_embed)
    input_ch = input_ch_s + input_ch_time
    output_ch = 1
    skips = [5]
    teacher = NeRF(D=args.netdepth, W=[128, 256, 256, 512, 256, 256, 256, 256],
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                spatial_dim=input_ch_s, temporal_dim=input_ch_time, hidden_dim=256, num_layers=8, region=args.N_samples_t).to(device)
    
    teacher = nn.DataParallel(teacher)
    teacher = teacher.cuda()
    ckpt_path = args.pretrain
    ckpt = torch.load(ckpt_path)
    teacher.load_state_dict(ckpt['network_fn_state_dict'])
    print(teacher)
    teacher.eval()
    test=True
    teacher_query_fn = lambda inputs, time_info, network_fn: run_network(inputs, time_info, network_fn, test=True,
                                                                         embed_fn=embed_fn, embedtime_fn=embedtime_fn, spatial_dim=input_ch_s, temporal_dim=input_ch_time)

    # Student model setting
    skips_s = [2]
    student = NeRF_student_3(D=args.netdepth, W=[128, 256, 256, 512, 256, 256, 256, 256],
                input_ch=input_ch, output_ch=output_ch, skips=skips_s,
                spatial_dim=input_ch_s, temporal_dim=input_ch_time, hidden_dim=128, num_layers=4, region=args.N_samples_s).to(device)
    
    student = nn.DataParallel(student)
    student = student.cuda()
    print(student)
    student.apply(init_weights)
    grad_vars = list(student.parameters())
    student_query_fn = lambda inputs, time_info, network_fn: run_network(inputs, time_info, network_fn, test=False,
                                                                          embed_fn=embed_fn, embedtime_fn=embedtime_fn, spatial_dim=input_ch_s, temporal_dim=input_ch_time)

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(.9, .999))
    start = 0
    basedir = args.basedir
    expname = args.expname

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
        student.load_state_dict(ckpt['network_fn_state_dict'])

    render_kwargs_train = {
        'network_query_fn' : student_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples_s, 
        'network_fn' : student,
    }
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    render_kwargs_teacher = {
        'network_query_fn' : teacher_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples_t, 
        'network_fn' : teacher,
    }

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, render_kwargs_teacher

# def render_only():
#     logging.info('Render Only!')
#     with torch.no_grad():
#         testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
#         os.makedirs(testsavedir, exist_ok=True)
#         print('Input Slice Shape', input_slices_imgs[...,0].shape)
        
#         feature_student, recon_volumes_img = render_path(D, H, W, input_test_slices_imgs, input_test_times, args.chunk, render_kwargs_test)
#         # time t_0
        
#         # time t_i
#         mae, psnr, ssim = test_metrics(recon_volumes_img, input_test_volumes_img.cpu().numpy())
#         print('MAE: {}, PSNR: {}, SSIM: {}'.format(mae, psnr, ssim))
#         recon_volumes_img_out = np.swapaxes(recon_volumes_img, 0, 3)
#         sitk.WriteImage(sitk.GetImageFromArray(
# recon_volumes_img_out), os.path.join(testsavedir, 'test_4d.nii.gz'))
#         print('Finish Reconstruction', testsavedir)

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, default='acdc_p003_student',
                        help='experiment name')
    parser.add_argument("--patient_id", type=int, default=3,
                        help='patient id')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='../dataset/mri/acdc', 
                        help='input data directory')
    parser.add_argument("--pretrain", type=str, default='./logs/acdc_p003/020000.tar', 
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
    parser.add_argument("--chunk", type=int, default=1024*8, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*2*2*2, 
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
    parser.add_argument("--N_samples_t", type=int, default=2*2*2, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_samples_s", type=int, default=4, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_neigh", type=int, default=10, 
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
    parser.add_argument("--i_weights", type=int, default=1000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_test", type=int, default=5000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--local-rank", default=-1, type=int)

    return parser

def train():
    parser = config_parser()
    args = parser.parse_args()

    # load data
    if args.dataset_type == 'mri':
        trainset, testset = load_mri_data_acdc(args.datadir, args.patient_id)
        volumes_img, volumes_k, slices_imgs, slices_k, times = trainset
        testvolumes_img, testvolumes_k, testslices_imgs, testslices_k, testtimes = testset
    total_N, D, H, W = volumes_k.shape[0], volumes_k.shape[1], volumes_k.shape[2], volumes_k.shape[3]

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

    # record config and log information
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    log_file = logging.FileHandler(filename=os.path.join(basedir, expname, 'logs.txt'), encoding='utf-8', mode='a+')
    logger.addHandler(log_file)

    # create teacher and student model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, render_kwargs_teacher = create_teacher_student_nerf(args)
    gloabl_step = start

    neighbor_dict = {
        'neigh': args.N_neigh,
    }
    render_kwargs_train.update(neighbor_dict)
    render_kwargs_test.update(neighbor_dict)

    if args.render_only:
        logging.info('Render Only!')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('Input Slice Shape', input_slices_imgs[...,0].shape)
            
            # new_input_test_slices_imgs = torch.tile(input_test_slices_imgs[0], (input_test_slices_imgs.shape[0], 1, 1, 1))
            # new_input_test_times = torch.squeeze(torch.tile(input_test_times[0], (1,20)))

            recon_volumes_img = render_path(D, H, W, input_test_slices_imgs, input_test_times, args.chunk, render_kwargs_test)
            recon_volumes_img = np.clip(recon_volumes_img, 0, 1)
            # time t_0
            
            # time t_i
            mae, psnr, ssim = test_metrics(recon_volumes_img, input_test_volumes_img.cpu().numpy())
            print('MAE: {}, PSNR: {}, SSIM: {}'.format(mae, psnr, ssim))
            recon_volumes_img_out = np.swapaxes(recon_volumes_img, 0, 3)
            sitk.WriteImage(sitk.GetImageFromArray(recon_volumes_img_out), os.path.join(testsavedir, 'test_4d.nii.gz'))
            print('Finish Reconstruction', testsavedir)
            # for i in range(volumes_img.shape[0]):
            #     sitk.WriteImage(sitk.GetImageFromArray(volumes_img[i]), os.path.join(testsavedir, str(i).zfill(3)+'.nii.gz'))
            # imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(show2ds), fps=30, quality=8)
        return 
    N_iters = 50010
    logging.info('Begin')
    i_batch = 0
    sw_loss = SlicedWassersteinLoss()
    l1char_loss = L1_Charbonnier_loss()
    for i in range(start, N_iters):
        # dataloader 
        i_volume_k_0 = input_volumes_k[i_batch]   # D, H, W
        i_volume_img_0 = input_volumes_img[i_batch]   # D, H, W
        i_slice_k_0 = input_slices_k[i_batch]     # H, W
        i_slice_img_0 = input_slices_imgs[i_batch]     # H, W
        i_time_0 = input_times[i_batch]       # 1

        di = np.floor(np.random.rand(1)[0] * (total_N-i_batch-1)).astype(np.int_)

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
        sample_p = torch.flatten(torch.ones_like(i_volume_img))
        diff = torch.abs((i_volume_img - i_volume_img_0).flatten())
        sample_p[diff>(diff.median()/2*3)] = 10
        sample_p = torch.softmax(torch.tensor(sample_p),-1)

        for epoch in range(num_epoch):
            select_inds = np.random.choice(coords_flat.shape[0], size=[args.N_rand], replace=False, p=sample_p.cpu().numpy()) # N_rand, 
            sample_p[select_inds] = -1
            sample_p = torch.softmax(torch.tensor(sample_p),-1)
            # select_inds = np.random.choice(coords_flat.shape[0], size=[args.N_rand], replace=False) # N_rand, 
            select_coords = coords_flat[select_inds].long() # N_rand, 3

            # time t_0
            coords_slice_0, coords_time_0 = get_coords_singleview(D, H, W, i_slice_img_0, i_time_0)
            pts_slice_0 = coords_slice_0[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]] # N_rand, 1 只有1维数据太少了，不足以反应三维coordinate
            pts_time_0 = coords_time_0[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]   # N_rand, 1
            batch_pts_0 = torch.stack([pts_slice_0, pts_time_0], 0) # 2, N_rand, 1
            target_s_k_0 = i_volume_k_0[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]
            target_s_img_0 = i_volume_img_0[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]
            # time t_i
            coords_slice, coords_time = get_coords_singleview(D, H, W, i_slice_img, i_time)            
            pts_slice = coords_slice[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]] # N_rand, 1 只有1维数据太少了，不足以反应三维coordinate
            pts_time = coords_time[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]   # N_rand, 1
            batch_pts = torch.stack([pts_slice, pts_time], 0) # 2, N_rand, 1
            target_s_k = i_volume_k[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]
            target_s_img = i_volume_img[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]

            # Get output from teacher
            # x_teacher0, t_teacher0, feature_teacher0, output_teacher0 = render_only(D, H, W, batch_pts_0, chunk=args.chunk, **render_kwargs_teacher)
            with torch.no_grad():
                x_teacher, t_teacher, feature_teacher, output_teacher = render(D, H, W, batch_pts, chunk=args.chunk, **render_kwargs_teacher)
            # Get output from student
            # x_student0, t_student0, feature_student0, output_student0 = render(D, H, W, batch_pts_0, chunk=args.chunk, **render_kwargs_train)
            x_student, t_student, feature_student, output_student = render(D, H, W, batch_pts, chunk=args.chunk, **render_kwargs_train)
            # training
            optimizer.zero_grad()
            # import pdb
            # pdb.set_trace()
            feat_loss = l1char_loss(feature_student, feature_teacher) #+ img2mse(x_student, x_teacher) + img2mse(t_student, t_teacher)
            # out_loss = img2mse(output_student, output_teacher) #+ 10*sw_loss(output_student, target_s_img)
            out_loss = img2mse(output_student, target_s_img) + 10*sw_loss(output_student, target_s_img)
            loss = out_loss + 2*feat_loss #+ 100*(torch.abs((output_student.max()-output_student.min())-(output_teacher.max()-output_teacher.min())))
            # loss = torch.abs((output_student.max()-output_student.min())-(output_teacher.max()-output_teacher.min()))
            
            # attention = torch.abs(target_s_img - target_s_img_0)
            # mask = torch.ones_like(target_s_img)
            # mask[attention>0.01] = 10
            # img_loss_0 = weighted_mse_loss(torch.squeeze(output_student0), target_s_img_0, mask) + 10 *sw_loss(torch.squeeze(output_student0), target_s_img_0) #+ ssim_loss(torch.squeeze(volume_img_0), target_s_img_0) 
            # img_loss = weighted_mse_loss(torch.squeeze(output_student), target_s_img, mask) + 10 *sw_loss(torch.squeeze(output_student), target_s_img) #+ ssim_loss(torch.squeeze(volume_img), target_s_img)
            # temporal_loss = l1char_loss((output_student - output_student0), (target_s_img - target_s_img_0)) 
            # loss = img_loss_0 + img_loss + temporal_loss

            psnr = mse2psnr(img2mse(output_student, target_s_img))
            mae = img2mae(output_student, target_s_img)
            loss.backward()
            optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (gloabl_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if (i)%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': gloabl_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
            }, path)
            logging.info('Saved checkpoints at {}'.format(path))
        
        # if i % args.i_test==0:
        #     render_only()

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
            logger.info(f"[TRAIN] Duration:{currenttime} Iter: {i} Loss: {loss.item()} PSNR: {psnr.item()} MAE: {mae.item()} Pred: {output_teacher.detach().cpu().numpy().max(), output_teacher.detach().cpu().numpy().mean()} GT: {target_s_img.mean()}")

        gloabl_step += 1

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()        