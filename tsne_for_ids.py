import torch
from torch import nn
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from reid.models.resnet_uncertainty import ResNetSimCLR
import glob
import re

# ========== 1. 模型加载函数 ==========
def load_model(weight_path, backbone, device='cuda'):
    model = backbone() # 加载模型结构
    ckpt = torch.load(weight_path, map_location='cpu') # 加载权重文件
    model.load_state_dict(ckpt['state_dict'], strict=False) # 加载模型参数
    model.to(device)
    model.eval() # 设置推理模型
    return model # 返回模型

# 路径提取函数
def get_image_paths(camera_id, person_id, dir_path="/data1/swx/datasets/market1501/Market-1501-v15.09.15/bounding_box_train"):
    img_paths = sorted(glob.glob(os.path.join(dir_path,'*.jpg')))
    pattern = re.compile(r'([-\d]+)_c(\d)')
    paths = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == person_id:
            if camid == camera_id:
                paths.append(img_path)
    
    return paths

# ========== 2. 特征提取函数 ==========
@torch.no_grad()
def extract_features(model, img_paths, transform, device='cuda'):
    feats = [] # 保存图片特征向量
    for p in img_paths:
        img = Image.open(p).convert('RGB') # 打开RGB图片，保证是三通道 
        x = transform(img).unsqueeze(0).to(device) # 对图片做标准预处理
        f = model(x) # 提取向量特征,返回一个元组
        f = f[0] # 取出第一个数据，即原型特征向量
        feats.append(f.cpu().numpy().squeeze()) # 存储结果
    feats = np.stack(feats) # 拼接成为二维矩阵
    return feats # 返回原型特征向量

# ========== 3. 原型计算函数 ==========
def compute_prototype(features):
    return np.mean(features, axis=0) # 返回同一个人的特征原型

# ========== 4. t-SNE 可视化函数 ==========
def visualize_tsne(features_dict, save_path):
    """
    features_dict: dict
       key: label string, e.g. "cam1_model_on_cam1"
       val: np.ndarray (N, D)
    """
    all_feats = [] # 存储特征矩阵
    all_labels = [] # 存储特征标签
    for k, v in features_dict.items():
        all_feats.append(v)
        all_labels += [k] * len(v)
    all_feats = np.concatenate(all_feats) # 拼接成为一个矩阵
    
    tsne = TSNE(n_components=2, perplexity=5, random_state=42) # 高维特征降维到二维
    feats_2d = tsne.fit_transform(all_feats) # 执行降维
    
    plt.figure(figsize=(8, 6))
    for label in set(all_labels):
        idx = [i for i, l in enumerate(all_labels) if l == label]
        plt.scatter(feats_2d[idx, 0], feats_2d[idx, 1], label=label, alpha=0.7)
    plt.legend()
    plt.title("t-SNE Visualization of Feature Drift Across Cameras")
    plt.tight_layout()
    plt.savefig(save_path,dpi=300,bbox_inches='tight')
    print(f"图片已保存到: {save_path}")

# ========== 5. 示例主流程 ==========
# 假设你有 get_image_paths(camera_id, person_id) 函数返回该 ID 图像路径
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

backbone = lambda: ResNetSimCLR(num_classes=751)  # 替换为实际模型定义

# 加载模型
model_cam0 = load_model('/data1/swx/Project/logs/try_without_global_prot/market1501cam_0_checkpoint.pth.tar', backbone)
model_cam1 = load_model('/data1/swx/Project/logs/try_without_global_prot/market1501cam_1_checkpoint.pth.tar', backbone)

# 提取特征
id = 2
cam0_imgs = get_image_paths(camera_id=1, person_id=id)
cam1_imgs = get_image_paths(camera_id=2, person_id=id)

f00 = extract_features(model_cam0, cam0_imgs, transform)
f10 = extract_features(model_cam1, cam0_imgs, transform)
f11 = extract_features(model_cam1, cam1_imgs, transform)

# 计算原型
proto11 = compute_prototype(f00)
proto21 = compute_prototype(f10)
proto22 = compute_prototype(f11)

# 组合成可视化输入
features_dict = {
    "cam0_model_cam0": f00,
    "cam1_model_cam0": f10,
    "cam1_model_cam1": f11
}

save_path = './id_0002_tsne.png'
visualize_tsne(features_dict,save_path=save_path)