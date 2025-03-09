from argparse import ArgumentParser

import utils
import torch
from models.basic_model import CDEvaluator

import os

"""
quick start

sample files in ./samples

save prediction files in the ./samples/predict

"""
import argparse
import os
from osgeo import gdal
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST



# 新增适配器类
class DualInputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x1 = x[:, :6, :, :]  # 时相A (假设输入通道为6)
        x2 = x[:, 6:, :, :]  # 时相B (假设输入通道为6)
        print(x1.shape)
        print()
        return self.model(x1, x2)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/B/2021-01-01_540400_55.png',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'fem', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam', 'shapleycam'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')



    # ------------
    # args
    # ------------
    parser.add_argument('--project_name', default='A:\Checkpoints/NBR_PLUS_VV_VH_RE4_S2_S1', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoint', type=str)
    parser.add_argument('--output_folder', default='samples/predict/Common_Files', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='quick_start', type=str)
    parser.add_argument('--image_pathA', default='/home/public/sch005406/Anjl/ChangeDetect/Original_Data/output/NBR_PLUS_VV_VH_RE4_S2_S1/A/20190108T022059_20190108T022054_T52TFM_20190106_3089679.tif', type=str)
    parser.add_argument('--image_pathB', default='/home/public/sch005406/Anjl/ChangeDetect/Original_Data/output/NBR_PLUS_VV_VH_RE4_S2_S1/B/20190108T022059_20190108T022054_T52TFM_20190106_3089679.tif', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="demo", type=str)
    parser.add_argument('--img_size', default=640, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    ''' base_transformer_pos_s4_dd8_dedim8'''
    parser.add_argument('--checkpoint_name', default='/home/public/sch005406/Anjl/ChangeDetect/BIT_CD_Tif/checkpoint/NBR_PLUS_VV_VH_RE4_S2_S1/best_ckpt.pt', type=str)

    args = parser.parse_args()

    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':

    args = get_args()
    utils.get_device(args)
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids)>0
                        else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.output_folder, exist_ok=True)

    log_path = os.path.join(args.output_folder, 'log_vis.txt')

    data_loader = utils.get_loader(args.data_name, img_size=args.img_size,
                                   batch_size=args.batch_size,
                                   split=args.split, is_train=False)

    model = CDEvaluator(args)
    # print(model.net_G)
    model.load_checkpoint(args.checkpoint_name)

    mpath = ''
    checkpoint = torch.load(args.checkpoint_name, map_location=device)
    # model.load_state_dict(checkpoint['model'])

    model.eval()


    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "fem": FEM,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM,
        'shapleycam': ShapleyCAM
    }

    if args.device == 'hpu':
        import habana_frameworks.torch.core as htcore


    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    # target_layers = [model]
    # target_layers = [model.net_G.resnet.layer4[1].bn2]
    target_layers = [model.net_G.resnet.layer4[1].conv2] 
    # 读取多波段图像A
    dataset_A = gdal.Open(args.image_pathA)
    img_A = dataset_A.ReadAsArray()

    # 读取多波段图像B
    dataset_B = gdal.Open(args.image_pathB)
    img_B = dataset_B.ReadAsArray()

    # 修改后的图像读取和预处理代码
    img_A = img_A[0:6].astype(np.float32) / 255.0  # 添加astype(np.float32)
    img_B = img_B[0:6].astype(np.float32) / 255.0  # 添加astype(np.float32)

    
    # 确保图像数据的形状是 (C, H, W)
    if img_A.ndim == 2:
        img_A = np.expand_dims(img_A, axis=0)
    # else:
    #     img_A = np.transpose(img_A, (2, 0, 1))  # 转置为 (C, H, W)
    if img_B.ndim == 2:
        img_B = np.expand_dims(img_B, axis=0)
    # else:
    #     img_B = np.transpose(img_B, (2, 0, 1))  # 转置为 (C, H, W)

    print(img_A.shape)


    img_A = np.transpose(img_A, (1, 2, 0))  # 转置为 (C, H, W)
    img_B = np.transpose(img_B, (1, 2, 0))  # 转置为 (C, H, W)
    print(img_A.shape)
    # rgb_imgA = cv2.imread(args.image_pathA, 1)[:, :, ::-1]
    # rgb_imgA = np.float32(rgb_imgA) / 255
    # rgb_imgB = cv2.imread(args.image_pathB, 1)[:, :, ::-1]
    # rgb_imgB = np.float32(rgb_imgB) / 255
    # 双时相输入处理（示例）
    rgb_img_A = preprocess_image(img_A, mean=[0.485, 0.456, 0.406,0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225,0.229, 0.224, 0.225])
    rgb_img_B = preprocess_image(img_B, mean=[0.485, 0.456, 0.406,0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225,0.229, 0.224, 0.225])



    
    input_tensor = torch.cat([rgb_img_A, rgb_img_B], dim=1).to(args.device)
    
    input_tensor = input_tensor.requires_grad_(True)  # 启用梯度

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputReST(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    # cam_algorithm = methods[args.method]
    # with cam_algorithm(model=model.net_G,  # 使用实际的PyTorch模型
    #                    target_layers=target_layers) as cam:


    wrapped_model = DualInputWrapper(model.net_G)
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=wrapped_model,
                      target_layers=target_layers) as cam:


        # 输入张量处理 (确保形状为 [1, 12, 640, 640])
        rgb_img_A = preprocess_image(img_A, mean=[0.485, 0.456, 0.406,0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225,0.229, 0.224, 0.225])  # 输入形状 [1,6,640,640]
        rgb_img_B = preprocess_image(img_B, mean=[0.485, 0.456, 0.406,0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225,0.229, 0.224, 0.225])  # 输入形状 [1,6,640,640]
        input_tensor = torch.cat([rgb_img_A, rgb_img_B], dim=1).to(args.device)

    
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        print(input_tensor.shape)
        # grayscale_cam = cam(input_tensor=input_tensor,
        #                     targets=targets,
        #                     aug_smooth=args.aug_smooth,
        #                     eigen_smooth=args.eigen_smooth)
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[lambda o: o.sum()],  # 直接对输出求和
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth
        )
        

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_imgA, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    os.makedirs(args.output_dir, exist_ok=True)

    cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
    gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
    cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')

    cv2.imwrite(cam_output_path, cam_image)
    cv2.imwrite(gb_output_path, gb)
    cv2.imwrite(cam_gb_output_path, cam_gb)







    for i, batch in enumerate(data_loader):
        name = batch['name']
        print('process: %s' % name)
        score_map = model._forward_pass(batch)
        model._save_predictions()







