import os
import torch
from utils import utils_image as util
from data.dataloder import Dataset as D
from torch.utils.data import DataLoader
from models.loss_vif import fusion_loss_vif as loss
import logging
from datetime import datetime
import time

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def logger(model_name, loss_value):
    log_dir = r".\distillation_test_log"

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"testing_{current_time}.log"

    logging.basicConfig(filename=os.path.join(log_dir, log_filename),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f"Model Name: {model_name}")
    for epoch, loss in enumerate(loss_value, start=1):
        logging.info(f"Epoch {epoch}, Loss: {loss}")
    logging.info(f"Average loss is {sum(loss_value) / len(loss_value)}")


def main():

    model = torch.load('./distillation_model/model.pth')
    model_name = 'distillation_Unet'
    loss_fn = loss()
    loss_log = []

    a_path = r'.\Dataset\testsets\MSRS\IR'
    b_path = r'.\Dataset\testsets\MSRS\VI_Y'
    test_set = D(a_path, b_path, 1)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    model.eval()
    for _, test_data in enumerate(test_loader):
        start = time.time()
        image_name = test_data['A_path'][0]
        img_a = test_data['A'].to(device)
        img_b = test_data['B'].to(device)

        window_size = 32
        with torch.no_grad():
            _, _, h_old, w_old = img_a.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :h_old + h_pad, :]
            img_a = torch.cat([img_a, torch.flip(img_a, [3])], 3)[:, :, :, :w_old + w_pad]
            img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :h_old + h_pad, :]
            img_b = torch.cat([img_b, torch.flip(img_b, [3])], 3)[:, :, :, :w_old + w_pad]
            outputs, _ = model(img_a, img_b)
        running_loss, _, _, _ = loss_fn(img_a, img_b, outputs)
        running_loss = running_loss.item()
        loss_log.append(running_loss)
        outputs = outputs[..., :h_old, :w_old]
        outputs = outputs.detach()[0].float().cpu()
        outputs = util.tensor2uint(outputs)
        save_dir = r'.\distillation_result'
        save_name = os.path.join(save_dir, os.path.basename(image_name))
        util.imsave(outputs, save_name)
        end = time.time()
        print(f'A picture is used {end - start}s')
    logger(model_name, loss_log)


if __name__ == '__main__':
    main()
