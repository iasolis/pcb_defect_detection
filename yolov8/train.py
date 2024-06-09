import argparse

import ultralytics.models
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, default='detect')
    parser.add_argument('--mode', required=True, default='train')
    parser.add_argument('--model', required=True, default='yolov8n.pt')
    parser.add_argument('--imgsz', required=True, default=1280)
    parser.add_argument('--data', required=True, default='datasets/data/data.yaml')
    parser.add_argument('--epochs', required=True, default=100)
    parser.add_argument('--batch', required=True, default=16)
    parser.add_argument('--name', required=True, default='yolov8s_100e')
    opt = parser.parse_args()

    model = YOLO(opt.model, task=opt.task, verbose=True)
    results = model.train(mode=opt.mode, data=opt.data, imgsz=int(opt.imgsz),
                          epochs=int(opt.epochs), batch=int(opt.batch), plots=True)


if __name__ == '__main__':
    main()
