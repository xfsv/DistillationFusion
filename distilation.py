import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import sys
from models.network_swinfusion1 import SwinFusion as net
from utils import utils_image as util
from data.dataloder import Dataset as D
from torch.utils.data import DataLoader
from models.loss_vif import fusion_loss_vif as loss
from Unet import UNet
import logging
import os
from datetime import datetime
import Total_loss
from Total_loss import TotalLoss
from MobileModule import MobileViT, model_config, Transformer

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

#################################
#   The Infrared
#   Version: With all layer distillation
#################################


def process_of_dim(image):
    image = np.squeeze(image)
    if image.ndim == 3:
        image = image[:, :, [2, 1, 0]]
    return image


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


def initial_the_teacher_fusion_model(args):

    # set up model
    model_path = os.path.join(args.model_path, args.iter_number + '_G.pth')  # 载入模型参数

    # 判断是否有预训练的参数，如果没有预训练参数直接停止程序
    if os.path.exists(model_path):
        print(f'loading model from {args.model_path}')
    else:
        print('Traget model path: {} not existing!!!'.format(model_path))
        sys.exit()

    model = define_model(args)  # 根据输入的基本参数确定模型
    model.eval()  # 将模型改为评估模式
    model = model.to(device)  # 把模型加入CUDA运算

    return model


def logger(model_name, loss_value):
    log_dir = r".\distillation_train_log"
    os.makedirs(log_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"training_{current_time}.log"

    logging.basicConfig(filename=os.path.join(log_dir, log_filename),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(f"Model Name: {model_name} Optimizer: Adam")
    for epoch, loss in enumerate(loss_value, start=1):
        logging.info(f"Epoch {epoch}, Loss: {loss}")


def train_knowledge_distillation(teacher, student, train_loader, epochs, optimizer, loss_fn):
    teacher.eval()
    student.train()

    loss_log = []
    model_name = 'Fusion without distillation'
    for epoch in range(epochs):
        running_loss = 0
        start = time.time()
        print(f"--------Epoch: {epoch+1:>3d}--------")
        mse_loss = nn.MSELoss()
        temp_img_name = 0
        temp_img_b = 0
        temp_img_a = 0
        for batch, train_data in enumerate(train_loader):
            optimizer.zero_grad()
            image_name = train_data['A_path'][0]
            img_a = train_data['A'].to(device)
            img_b = train_data['B'].to(device)

            if epoch % 10 == 0:
                temp_img_a = img_a
                temp_img_b = img_b
                temp_img_name = image_name

            # with torch.no_grad():
            #     output_teacher, feature_map_teacher = teacher(img_a, img_b)
            #     output_teacher = output_teacher.float().detach()
            output_student = student(img_a, img_b)
            loss_student, _, _, _ = loss_fn(img_a, img_b, output_student)
            # output_student = output_student.float()
            # loss_last_layer = TotalLoss()
            # loss_student_teacher = loss_last_layer(output_student, output_teacher)
            loss_total = torch.tensor(0).float().to(device)
            loss_total += loss_student.to(device)
            loss_total.backward()  # 这个地方一定要注意！！！！
            optimizer.step()

            # running_loss += loss_total
            running_loss += loss_total.item()

        end = time.time()
        if epoch % 10 == 0:
            outputs = student(temp_img_a, temp_img_b)
            outputs = outputs.detach()[0].float().cpu()
            outputs = util.tensor2uint(outputs)
            save_dir = r'.\distillation_result'
            save_name = os.path.join(save_dir, os.path.basename(temp_img_name))
            util.imsave(outputs, save_name)
        loss_log.append(running_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss} ,Used time: {end - start}")
    logger(model_name=model_name, loss_value=loss_log)
    model = student
    torch.save(model, './distillation_model/model.pth')


def output_the_result(model, inputs, img_name):
    outputs = model(inputs)
    outputs = outputs.detach().float()
    outputs = util.tensor2uint(outputs)
    save_dir = r'.\distillation_result'
    save_name = os.path.join(save_dir, os.path.basename(img_name))
    util.imsave(outputs, save_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fusion', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='./Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/')
    parser.add_argument('--iter_number', type=str,
                        default='10000')
    parser.add_argument('--root_path', type=str, default='./Dataset/trainsets/',
                        help='input test image root folder')
    parser.add_argument('--dataset', type=str, default='MSRS',
                        help='input test image name')
    parser.add_argument('--A_dir', type=str, default='IR',
                        help='input test image name')
    parser.add_argument('--B_dir', type=str, default='VI_Y',
                        help='input test image name')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
    args = parser.parse_args()

    teacher = initial_the_teacher_fusion_model(args)

    config = model_config.get_config("xx_small")
    original_model = MobileViT.MobileViT(config, num_classes=1000).to(device)
    weight_path = r'\MobileViT\mobilevit_xxs.pt'
    weights_dict = torch.load(weight_path, map_location=device)
    # 删除有关分类类别的权重
    for k in list(weights_dict.keys()):
        if "classifier" in k:
            del weights_dict[k]
    original_model.load_state_dict(weights_dict, strict=False)
    for name, para in original_model.named_parameters():
        if "MobileViT" in name:
            para.requires_grad_(False)
    student = UNet(original_model).to(device)

    a_dir = os.path.join(args.root_path, args.dataset, args.A_dir)
    b_dir = os.path.join(args.root_path, args.dataset, args.B_dir)

    train_set = D(a_dir, b_dir, args.in_channel)
    train_loader = DataLoader(train_set, batch_size=16,
                              shuffle=True, num_workers=4,
                              drop_last=True, pin_memory=False)

    epochs = 500
    loss_fn = loss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    train_knowledge_distillation(teacher=teacher, student=student, train_loader=train_loader,
                                 epochs=epochs, optimizer=optimizer, loss_fn=loss_fn)


if __name__ == '__main__':
    main()
