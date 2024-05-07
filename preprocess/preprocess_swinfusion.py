import argparse
from cgi import test
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import time
import sys
from models.network_swinfusion1 import SwinFusion as net
from utils import utils_image as util
from data.dataloder import Dataset as D
from torch.utils.data import DataLoader
from models import loss_vif

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fusion', help='classical_sr, lightweight_sr, real_sr, '
                                                                   'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='../Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/')
    parser.add_argument('--iter_number', type=str,
                        default='10000')
    parser.add_argument('--root_path', type=str, default='../Dataset/trainsets/',
                        help='input test image root folder')
    parser.add_argument('--dataset', type=str, default='MSRS',
                        help='input test image name')
    parser.add_argument('--A_dir', type=str, default='IR',
                        help='input test image name')
    parser.add_argument('--B_dir', type=str, default='VI_Y',
                        help='input test image name')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model_path = os.path.join(args.model_path, args.iter_number + '_G.pth')
    if os.path.exists(model_path):
        print(f'loading model from {args.model_path}')
    else:
        print('Traget model path: {} not existing!!!'.format(model_path))
        sys.exit()
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    a_dir = os.path.join(args.root_path, args.dataset, args.A_dir)
    b_dir = os.path.join(args.root_path, args.dataset, args.B_dir)
    os.makedirs(save_dir, exist_ok=True)
    train_set = D(a_dir, b_dir, args.in_channel)
    test_loader = DataLoader(train_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    feature_map_list = []
    for i, test_data in enumerate(test_loader):
        imgname = test_data['A_path'][0]
        img_a = test_data['A'].to(device)
        img_b = test_data['B'].to(device)
        start = time.time()
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_a.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :h_old + h_pad, :]
            img_a = torch.cat([img_a, torch.flip(img_a, [3])], 3)[:, :, :, :w_old + w_pad]
            img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :h_old + h_pad, :]
            img_b = torch.cat([img_b, torch.flip(img_b, [3])], 3)[:, :, :, :w_old + w_pad]
            output, feature_map = te1st(img_a, img_b, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]
            output = output.detach()[0].float().cpu()
            feature_map = feature_map.detach()[0].float().cpu()
            feature_map_list.append(feature_map)
        end = time.time()
        output = util.tensor2uint(output)
        save_name = os.path.join(save_dir, os.path.basename(imgname))
        util.imsave(output, save_name)
        print(
            "[{}/{}]  Saving fused image to : {}, Processing time is {:4f} s".format(i + 1, len(test_loader), save_name,
                                                                                     end - start))
    save_feature_map_name = os.path.join(save_dir, 'feature_maps.pt')
    torch.save(feature_map_list, save_feature_map_name)

def define_model(args):
    model = net(upscale=args.scale, in_chans=args.in_channel, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler=None, resi_connection='1conv')
    param_key_g = 'params'
    model_path = os.path.join(args.model_path, args.iter_number + '_E.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)

    return model


def setup(args):
    save_dir = f'../Dataset/trainsets/MSRS/SwinFusion_{args.dataset}_process'
    folder = os.path.join(args.root_path, args.dataset, 'A_Y')
    print('folder:', folder)
    border = 0
    window_size = 8

    return folder, save_dir, border, window_size


def te1st(img_a, img_b, model, args, window_size):
    # test the image as a whole
    output, feature_map = model(img_a, img_b)

    return output, feature_map


if __name__ == '__main__':
    main()