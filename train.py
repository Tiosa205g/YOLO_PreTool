from ultralytics import YOLO
import torch
import os
from c2net.context import prepare,upload_output

#初始化导入数据集和预训练模型到容器内
c2net_context = prepare()

#获取数据集路径
brain_tumor_path = c2net_context.dataset_path+"/"+"brain-tumor"

#获取预训练模型路径
yolo26m_path = c2net_context.pretrain_model_path+"/"+"yolo26m"


#输出结果必须保存在该目录
you_should_save_here = c2net_context.output_path

#回传结果到openi，只有训练任务才能回传
upload_output()

RESUME = False

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    print(brain_tumor_path,yolo26m_path)
    print(os.listdir(brain_tumor_path),os.listdir(yolo26m_path))
    # Load a model
    model = YOLO(yolo26m_path+"/yolo26m.pt")  # load a pretrained model (recommended for training)
    #model = YOLO('best.pt')
    results = model.train(data="data.yaml",
                          epochs=100,
                          imgsz=640,
                          device=device,
                          resume = RESUME
                          )