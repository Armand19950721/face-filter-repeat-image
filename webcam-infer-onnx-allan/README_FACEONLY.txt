# 環境設定
python=3.8
cuda=11.8
pip install onnx onnxruntime-gpu==1.14.1 opencv-python

# 執行 "僅臉部關鍵點"
python webcam_infer_onnx_FACEONLY.py

# 執行 "臉部關鍵點 & 髮色" (較耗效能)
python webcam_infer_onnx_FACEONLY.py --face_only 0