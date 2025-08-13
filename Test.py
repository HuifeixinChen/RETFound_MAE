import os, types, cv2, torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import timm
from timm.models.vision_transformer import Attention as TimmAttention
from models_vit import RETFound_mae

# ===================== CONFIG =====================
BACKBONE = "vit"            # "vit" 或 "resnet"
MODEL_NAME = "RETFound_mae" # 仅用于文件命名展示
MODEL_PATH = r"F:\retfound_papil_finetune\RETFound_MAE\output_dir\RETFound_mae_natureCFP_Papil\checkpoint_best.pth"
IMG_DIR    = r"F:\retfound_papil_finetune\RETFound_MAE\data\test\Unknow"
NB_CLASSES = 3
CLASS_NAMES = ["normal", "papilledema", "Pseudopapilledema"]

# ViT Rollout 可调参数
ROLL_DISCARD = 0.3          # 丢弃小注意力比例 0~0.6
ROLL_LAST_K  = 6            # 只用最后K层做rollout，0或None表示用全部层
HEAD_FUSION  = "max"        # "mean" 或 "max"

# 可视化强度
IMAGE_WEIGHT = 0.45         # 原图权重（0~1），热图权重=1-IMAGE_WEIGHT

# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出目录
model_dir = os.path.dirname(MODEL_PATH)
csv_path  = os.path.join(model_dir, f"predictions_no_label_{MODEL_NAME}.csv")
save_dir_cam  = os.path.join(model_dir, f"gradcam_results_{MODEL_NAME}")
save_dir_roll = os.path.join(model_dir, f"rollout_results_{MODEL_NAME}")
save_dir_trip = os.path.join(model_dir, f"triptych_{MODEL_NAME}")
os.makedirs(save_dir_cam, exist_ok=True)
os.makedirs(save_dir_roll, exist_ok=True)
os.makedirs(save_dir_trip, exist_ok=True)

# 预处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

# ---------- 构建模型 ----------
if BACKBONE.lower() == "vit":
    model = RETFound_mae(img_size=224, num_classes=NB_CLASSES, global_pool=True)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # patch & grid
    patch_h, patch_w = model.patch_embed.patch_size
    H, W = 224 // patch_h, 224 // patch_w
    print(f"[INFO] ViT patch grid: {H}x{W}")

    # 打补丁：保存 softmax 后注意力
    def _attn_forward_with_save(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        self.last_attn = attn.detach()
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x

    for blk in model.blocks:
        if isinstance(blk.attn, TimmAttention) and not hasattr(blk.attn, "_patched_with_save"):
            blk.attn.forward = types.MethodType(_attn_forward_with_save, blk.attn)
            blk.attn._patched_with_save = True

    # Grad-CAM++ reshape（ViT）
    def vit_reshape_transform(tensor):
        feat = tensor[:,1:,:].reshape(tensor.size(0), H, W, tensor.size(2))
        return feat.permute(0,3,1,2)

    # 目标层：最后一个 block 的 norm1（稳定）
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=vit_reshape_transform)

elif BACKBONE.lower() == "resnet":
    # 用 timm resnet50
    model = timm.create_model('resnet50', pretrained=False, num_classes=NB_CLASSES)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # 目标层：最后一块的最后一层卷积
    target_layers = [model.layer4[-1].conv3]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

else:
    raise ValueError("BACKBONE 只能是 'vit' 或 'resnet'")

# ---------- Attention Rollout（仅ViT） ----------
def attention_rollout_from_model(model, H, W, discard_ratio=0.0, last_k=None, head_fusion="mean"):
    blocks = model.blocks if (not last_k or last_k<=0) else model.blocks[-last_k:]
    attns = []
    for blk in blocks:
        A = getattr(blk.attn, "last_attn", None)   # [B, heads, T, T]
        if A is None:
            raise RuntimeError("未找到 blk.attn.last_attn；请确认Attention已打补丁。")
        if head_fusion == "max":
            a = A.max(dim=1).values
        else:
            a = A.mean(dim=1)                      # [B, T, T]

        I = torch.eye(a.size(-1), device=a.device).unsqueeze(0)
        a = a + I
        a = a / a.sum(dim=-1, keepdim=True)

        if discard_ratio > 0:
            B, T, _ = a.shape
            flat = a.view(B, -1)
            k = int(discard_ratio * flat.shape[1])
            if k > 0:
                _, idxs = torch.topk(flat, k, largest=False)
                flat.scatter_(1, idxs, 0)
                a = flat.view(B, T, T)
        attns.append(a)

    R = attns[0]
    for i in range(1, len(attns)):
        R = R @ attns[i]

    cls = R[:,0,1:].reshape(-1, H, W)
    # 归一化
    cls = cls - cls.amin(dim=(1,2), keepdim=True)
    cls = cls / (cls.amax(dim=(1,2), keepdim=True) + 1e-8)
    return cls

# ---------- 可视化工具 ----------
def overlay_heatmap(rgb_img01, heat01, image_weight=IMAGE_WEIGHT):
    heat01 = np.clip(heat01, 0, 1)
    # 百分位裁剪 + gamma 提升对比
    lo, hi = np.percentile(heat01, [2, 98])
    heat01 = np.clip((heat01 - lo) / (hi - lo + 1e-8), 0, 1) ** 0.75
    color = cv2.applyColorMap(np.uint8(255 * heat01), cv2.COLORMAP_JET)
    base = np.uint8(rgb_img01 * 255)
    return cv2.addWeighted(base, image_weight, color, 1 - image_weight, 0)

def make_triptych(img224_rgb, cam_img, roll_img=None):
    h, w, _ = img224_rgb.shape
    pad = 8
    if roll_img is None:
        canvas = np.ones((h, w*2+pad, 3), dtype=np.uint8)*255
        canvas[:, :w] = img224_rgb
        canvas[:, w+pad:w*2+pad] = cam_img
    else:
        canvas = np.ones((h, w*3+2*pad, 3), dtype=np.uint8)*255
        canvas[:, :w] = img224_rgb
        canvas[:, w+pad:w*2+pad] = cam_img
        canvas[:, 2*w+2*pad:3*w+2*pad] = roll_img
    return canvas

# ---------- 推理 + 保存 ----------
rows = []
for fname in os.listdir(IMG_DIR):
    if not fname.lower().endswith((".jpg",".jpeg",".png",".tif",".bmp")):
        continue
    path = os.path.join(IMG_DIR, fname)
    img_pil = Image.open(path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 前向
    logits = model(img_tensor)
    probs = F.softmax(logits, dim=1)[0]
    pred_cls  = int(torch.argmax(probs))
    pred_conf = float(probs[pred_cls])

    rows.append({
        "model_name": MODEL_NAME,
        "image": path,
        "pred_class": pred_cls,
        "pred_name": CLASS_NAMES[pred_cls],
        "pred_confidence": pred_conf,
        "probabilities": probs.detach().cpu().numpy().tolist()
    })

    # 原图 224
    rgb224 = np.array(img_pil.resize((224,224)), dtype=np.float32)/255.0

    # Grad-CAM++
    grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred_cls)])[0]
    cam_overlay = show_cam_on_image(rgb224, grayscale_cam, use_rgb=True)
    cv2.putText(cam_overlay, f"{MODEL_NAME}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(cam_overlay, f"{CLASS_NAMES[pred_cls]} ({pred_conf*100:.1f}%)", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imwrite(os.path.join(save_dir_cam, fname), cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

    roll_overlay = None
    if BACKBONE.lower() == "vit":
        try:
            roll = attention_rollout_from_model(model, H, W, discard_ratio=ROLL_DISCARD, last_k=ROLL_LAST_K, head_fusion=HEAD_FUSION)[0]
            roll = cv2.resize(roll.detach().cpu().numpy(), (224,224))
            roll_overlay = overlay_heatmap(rgb224, roll, image_weight=IMAGE_WEIGHT)
            cv2.putText(roll_overlay, f"{MODEL_NAME} (Rollout)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(roll_overlay, f"{CLASS_NAMES[pred_cls]} ({pred_conf*100:.1f}%)", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imwrite(os.path.join(save_dir_roll, fname), cv2.cvtColor(roll_overlay, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"[WARN] Rollout失败（{fname}）：{e}")

    # 三联图
    trip = make_triptych(np.uint8(rgb224*255), np.uint8(cam_overlay), None if roll_overlay is None else np.uint8(roll_overlay))
    trip_path = os.path.join(save_dir_trip, fname)
    cv2.imwrite(trip_path, cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))

# CSV
pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
print(f"[INFO] CSV -> {csv_path}")
print(f"[INFO] Grad-CAM++ -> {save_dir_cam}")
print(f"[INFO] Rollout -> {save_dir_roll if BACKBONE=='vit' else '(ResNet无Rollout)'}")
print(f"[INFO] Triptych -> {save_dir_trip}")
