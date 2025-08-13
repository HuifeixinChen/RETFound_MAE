import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 导入你微调时用的模型定义
import models_vit as models

def load_model(checkpoint_path, device, nb_classes=3, input_size=224):
    """
    加载与微调时一致的 RETFound_MAE (ViT) 模型，并载入权重
    """
    model = models.RETFound_mae(
        img_size=input_size,
        num_classes=nb_classes,
        drop_path_rate=0.2,
        global_pool=True
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def main():
    # 1. 环境与路径设置
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = r'.\output_dir\RETFound_mae_natureCFP_Papil\checkpoint_best.pth'
    data_dir   = r'.\data\test'
    out_dir    = r'.\heatmaps'
    os.makedirs(out_dir, exist_ok=True)

    # 2. 图像预处理：两个 transform
    # a) 用于模型输入（包含归一化）
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # b) 仅缩放与转 Tensor，用于显示原图
    resize_only = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # 3. 加载微调后的模型
    model = load_model(checkpoint, device, nb_classes=3, input_size=224)

    # 4. 指定目标层：ViT 的 patch 嵌入投影层
    target_layer = model.patch_embed.proj
    cam = GradCAM(model=model, target_layers=[target_layer])

    # 5. 遍历每个类别下的前 5 张图生成热图
    for cls in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        for img_name in os.listdir(cls_dir)[:5]:
            img_path = os.path.join(cls_dir, img_name)
            img = Image.open(img_path).convert('RGB')

            # a) 模型输入
            inp = preprocess(img).unsqueeze(0).to(device)

            # b) 计算灰度热图
            grayscale_cam = cam(input_tensor=inp)[0]

            # c) 准备原图（缩放到 224x224 并转 Tensor）
            orig = resize_only(img).permute(1,2,0).numpy()
            orig = (orig - orig.min()) / (orig.max() - orig.min())

            # d) 叠加热图并保存
            heatmap = show_cam_on_image(orig, grayscale_cam, use_rgb=True)
            out_path = os.path.join(out_dir, f'{cls}_{img_name}')
            plt.imsave(out_path, heatmap)
            print(f'已保存热图: {out_path}')

if __name__ == '__main__':
    main()
