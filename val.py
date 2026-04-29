from ultralytics import YOLO
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('best.pt')
    res = model.val(device = device)
    print(res)
