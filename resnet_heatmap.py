import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import timm

def load_model(checkpoint_path, device, nb_classes=3, input_size=224):
    """
    加载 ResNet50 模型，并载入权重
    """
    # 用 timm 创建一个 resnet50
    model = timm.create_model(
        'resnet50',
        pretrained=False,
        num_classes=nb_classes,
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def main():
    # 设置
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = r'.\output_dir\resnet50_natureCFP_Papil\checkpoint_best.pth'
    data_dir   = r'.\data\test'
    out_dir    = r'.\heatmaps_resnet50'
    os.makedirs(out_dir, exist_ok=True)

    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    resize_only = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # 加载模型
    model = load_model(checkpoint, device, nb_classes=3, input_size=224)

    # ResNet50 最后一层卷积
    target_layer = model.layer4[-1].conv3
    cam = GradCAM(model=model, target_layers=[target_layer])

    # 遍历每个类别前 5 张图
    for cls in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        for img_name in os.listdir(cls_dir)[:5]:
            img_path = os.path.join(cls_dir, img_name)
            img = Image.open(img_path).convert('RGB')

            inp = preprocess(img).unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=inp)[0]

            orig = resize_only(img).permute(1,2,0).numpy()
            orig = (orig - orig.min()) / (orig.max() - orig.min())

            heatmap = show_cam_on_image(orig, grayscale_cam, use_rgb=True)
            out_path = os.path.join(out_dir, f'{cls}_{img_name}')
            plt.imsave(out_path, heatmap)
            print(f'已保存热图: {out_path}')

if __name__ == '__main__':
    main()
