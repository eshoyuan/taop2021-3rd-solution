import torch
import torch.nn as nn
import numpy as np
import cv2
import os

def inv_transform(x):
    x = ((x * 0.5) + 0.5) * 255.0
    return x

def image_save_coarse(save_path, image, prob_map, batch_list, batch_idx, batch_size):
    # print(image.size())
    save_img = image.cpu().detach().numpy()
    # print(save_img.max())
    ori_size_H, ori_size_W = image.size()[2], image.size()[3]
    # print(prob_map.size())
    save_mask = prob_map.cpu().detach().numpy().squeeze()
    # print(save_mask.max())

    for i in range(batch_size):
        # save_base_name = '{}_{}_'.format(batch_idx, i)
        save_base_name = os.path.basename(batch_list[i])[:-4]
        save_img_name = os.path.join(save_path, save_base_name + '_img.jpg')
        save_mask_name = os.path.join(save_path, save_base_name + '_mask.jpg')
        # save_gt_name = os.path.join(save_path, save_base_name + 'gt.jpg')
        cv2.imwrite(save_img_name, cv2.cvtColor(save_img.transpose((0, 2, 3, 1))[i, :, :, :].squeeze()*(255.0), cv2.COLOR_RGB2BGR))
        save_map = (save_mask[i, :, :].squeeze()*255.0).astype(np.uint8)
        # print(save_map)
        cv2.imwrite(save_mask_name, cv2.applyColorMap(cv2.resize(save_map, (ori_size_H, ori_size_W)), cv2.COLORMAP_JET))
        # cv2.imwrite(save_gt_name, save_gt.transpose((0, 2, 3, 1))[i, :, :, :].squeeze()*(255.0))