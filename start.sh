python3 -m venv ai_hack
source ai_hack/bin/activate
pip install -r requirements.txt
# python yolov7/detect.py --weights yolov7-tiny.pt --conf 0.25 --img-size 640 --source yolov7/inference/images/horses.jpg
# python yolov7/export.py --weights yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
