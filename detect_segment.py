"""
基于现有 YOLOv5 项目环境的 YOLO+U-Net 串联推理（含汇报用流程可视化）
放在 YOLOv5 项目根目录下运行
"""
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ---------- 直接引用 YOLOv5 的现成模块 ----------
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes, LOGGER, check_img_size

# ===================== 配置 =====================
YOLO_WEIGHTS   = 'weights/best.pt'           # 你的 YOLO 权重路径
UNET_WEIGHTS   = 'weights/best_unet.pth'     # U-Net 权重路径
DATA_YAML      = 'data/mydata.yaml'          # 数据集配置文件（包含类别名）
SOURCE         = 'data/images'               # 输入图像（单张或文件夹）
OUTPUT_DIR     = 'segmentation_output'       # 输出目录
IMG_SIZE       = 1056                        # 期望推理尺寸（自动调整为 stride 倍数）
CONF_THRES     = 0.3
IOU_THRES      = 0.45
EXPAND_RATIO   = 0.4                         # 外扩比例
UNET_INPUT_SIZE = 256
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 可视化设置（汇报用）
ENABLE_VIS = True            # 是否生成流程可视化图
NUM_VIS_SAMPLES = 0        # 最多为前几张图像生成可视化
# =================================================

# 读取类别名
with open(DATA_YAML, 'r', encoding='utf-8') as f:
    names = yaml.safe_load(f)['names']

# 加载 YOLO 模型
yolo_model = DetectMultiBackend(YOLO_WEIGHTS, device=DEVICE)
stride = yolo_model.stride
imgsz = check_img_size(IMG_SIZE, s=stride)
print(f'实际推理尺寸: {imgsz}')

# 加载 U-Net 模型
unet = smp.Unet('resnet18', encoder_weights=None, classes=1).to(DEVICE)
unet.load_state_dict(torch.load(UNET_WEIGHTS, map_location=DEVICE))
unet.eval()

# U-Net 预处理
unet_transform = A.Compose([
    A.Resize(UNET_INPUT_SIZE, UNET_INPUT_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def expand_bbox(xyxy, ratio, img_shape):
    """外扩边界框，不超出图像"""
    x1, y1, x2, y2 = xyxy
    h, w = img_shape[:2]
    dw = (x2 - x1) * ratio
    dh = (y2 - y1) * ratio
    x1 = max(0, int(x1 - dw))
    y1 = max(0, int(y1 - dh))
    x2 = min(w, int(x2 + dw))
    y2 = min(h, int(y2 + dh))
    return x1, y1, x2, y2

def process_mask(mask):
    """后处理：闭运算 + 填孔 + 最大连通域"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    cnt = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [cnt], -1, 255, cv2.FILLED)
    return cleaned

def make_visualization(original_img, yolo_boxes_img, crop_img, unet_mask, clean_mask, final_result, output_path):
    """
    生成流程可视化子图（2×3 布局）：
    1. 原图
    2. YOLO 检测框
    3. 裁剪子图
    4. U-Net 原始输出 (概率图热力图)
    5. 后处理掩膜
    6. 最终结果（原图+分割轮廓）
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    # 1. 原图
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('1. Original Image')
    axes[0].axis('off')

    # 2. YOLO 检测框
    axes[1].imshow(cv2.cvtColor(yolo_boxes_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('2. YOLO Detection Boxes')
    axes[1].axis('off')

    # 3. 裁剪子图
    axes[2].imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('3. Cropped Patch')
    axes[2].axis('off')

    # 4. U-Net 原始输出
    axes[3].imshow(unet_mask, cmap='hot')
    axes[3].set_title('4. U-Net Raw Output')
    axes[3].axis('off')

    # 5. 后处理掩膜
    axes[4].imshow(clean_mask, cmap='gray')
    axes[4].set_title('5. Post-processed Mask')
    axes[4].axis('off')

    # 6. 最终输出
    axes[5].imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
    axes[5].set_title('6. Final Output')
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info(f'流程可视化已保存: {output_path}')

# ---------- 主推理 ----------
dataset = LoadImages(SOURCE, img_size=imgsz, stride=stride, auto=yolo_model.pt)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
vis_dir = Path(OUTPUT_DIR) / 'vis'
vis_dir.mkdir(parents=True, exist_ok=True)

vis_count = 0

# ---------- 修复后的主推理循环片段 ----------
for path, im, im0s, _, _ in dataset:
    original_im = im0s.copy()  # 原始大图
    # 准备一个只画 YOLO 框的画布，不带分割轮廓
    yolo_vis = original_im.copy()

    im = torch.from_numpy(im).to(DEVICE).float() / 255.0
    if len(im.shape) == 3:
        im = im[None]

    pred = yolo_model(im)
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, agnostic=True)
    det = pred[0]

    vis_data = None

    if len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

        # --- 步骤 2 修正：在全图上绘制明显的检测框（加粗线条，方便看清） ---
        for *xyxy, conf, cls in det:
            x1_b, y1_b, x2_b, y2_b = map(int, xyxy)
            # 使用醒目的红色，并加粗线条 (thickness=5)
            cv2.rectangle(yolo_vis, (x1_b, y1_b), (x2_b, y2_b), (0, 0, 255), 5)
            cv2.putText(yolo_vis, f'Gage {conf:.2f}', (x1_b, y1_b - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        for *xyxy, conf, cls in reversed(det):
            cls_id = int(cls)
            cls_name = names[cls_id] if cls_id < len(names) else f'class_{cls_id}'

            # 外扩并裁剪
            x1, y1, x2, y2 = expand_bbox(xyxy, EXPAND_RATIO, im0s.shape)
            # --- 步骤 3 修正：裁剪“纯净”的子图（来自 original_im，不带轮廓） ---
            crop_pure = original_im[y1:y2 + 1, x1:x2 + 1].copy()

            if crop_pure.size == 0:
                continue

            # U-Net 分割逻辑 (保持不变)
            img_rgb = cv2.cvtColor(crop_pure, cv2.COLOR_BGR2RGB)
            inp = unet_transform(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mask_pred = unet(inp)[0, 0].cpu().numpy()
            mask_bin = (mask_pred > 0.5).astype(np.uint8) * 255####可以修改调高可能可以避免误识别
            mask_clean = process_mask(mask_bin)

            # 缩放并画到 im0s 上（最终结果图）
            mask_orig = cv2.resize(mask_clean, (x2 - x1 + 1, y2 - y1 + 1), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                cnt_global = cnt + np.array([x1, y1])
                cv2.drawContours(im0s, [cnt_global], -1, (0, 255, 0), 2)

            # --- 步骤 4/5 修正：可视化数据收集 ---
            if vis_data is None:
                # 概率热力图缩放到裁剪图大小
                unet_raw_display = cv2.resize(mask_pred, (crop_pure.shape[1], crop_pure.shape[0]))
                # 依次传入：原图, 带红框的全图, 纯净裁剪图, 热力图, 掩膜, 最终带绿轮廓的图
                vis_data = (original_im, yolo_vis, crop_pure, unet_raw_display, mask_clean, im0s.copy())

    # 生成流程可视化（仅前 NUM_VIS_SAMPLES 张有效结果）
    if ENABLE_VIS and vis_count < NUM_VIS_SAMPLES and vis_data is not None:
        vis_count += 1
        vis_output_path = vis_dir / f'{Path(path).stem}_pipeline.jpg'
        make_visualization(*vis_data, str(vis_output_path))

    # 保存最终结果
    out_path = str(Path(OUTPUT_DIR) / Path(path).name)
    cv2.imwrite(out_path, im0s)
    LOGGER.info(f'结果已保存: {out_path}')

print('推理完成，结果保存在:', OUTPUT_DIR)
print('流程可视化保存在:', str(vis_dir))