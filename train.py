import argparse
import torch
from pathlib import Path
import os
from c2net.context import prepare, upload_output
import shutil
import importlib
import subprocess
import sys


def install_package(package_name):
    """自动安装Python库"""
    try:
        importlib.import_module(package_name)
        print(f"✅ 库 {package_name} 已安装")
    except ImportError:
        print(f"🔍 未找到 {package_name}，正在自动安装...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                package_name,
                "--quiet",
                "--no-warn-script-location",
            ]
        )
        print(f"✅ {package_name} 安装完成")


# ====================== 自动检查并安装 ultralytics ======================
install_package("ultralytics")
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="yolo26m/yolo26m.pt")
parser.add_argument("--data", type=str, default="brain-tumor")
parser.add_argument("--data_argument", type=str, default="data.yaml")
parser.add_argument("--count", type=int, default=100)
args = parser.parse_args()
print(args)
# 初始化导入数据集和预训练模型到容器内
c2net_context = prepare()


# 获取数据集路径
brain_tumor_path = c2net_context.dataset_path + "/" + args.data
# 获取预训练模型路径
model_path = c2net_context.pretrain_model_path + "/" + args.model
file_path = Path(args.data_argument)
if file_path.is_file():
    root = Path(__file__).parent
    shutil.copy(file_path, root / file_path.name)

RESUME = False

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(brain_tumor_path, model_path)
    # Load a model
    model = YOLO(model_path)
    results = model.train(
        data=file_path.name,
        epochs=args.count,
        imgsz=640,
        device=device,
        resume=RESUME,
    )

    try:
        shutil.copy("runs/detect/train/weights/best.pt", c2net_context.output_path)
        # 回传结果到openi，只有训练任务才能回传
        upload_output()
        print("复制成功")
    except Exception as e:
        print(f"复制文件出错:{e}")
