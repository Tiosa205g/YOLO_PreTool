from ultralytics import YOLO
from PIL import Image
import torch

detect_files = ['test.jpg']

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('best.pt')
    res = model.predict(detect_files,
                        device = device,
                        retina_masks = True,
                        iou = 0.7,
                        agnostic_nms=True)
    for r in res:
        r.show()