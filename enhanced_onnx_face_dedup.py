'''
增強型ONNX人臉去重程序 (Enhanced ONNX Face Deduplication)
======================================================

本程序是一個基於ONNX模型的人臉檢測與特徵提取系統，專門用於處理難以識別的人臉圖像。
此版本專注於提高特徵提取成功率，實現了多種圖像增強技術。

特徵提取增強技術:
--------------
1. 多角度嘗試：在特徵提取前對裁剪區域進行輕微旋轉（±15°，±30°）
2. 多種圖像增強組合：亮度調整、對比度增強、高斯模糊、直方圖均衡化等
3. 人臉對齊：使用檢測到的關鍵點進行面部對齊，再提取特徵
4. 擴大邊界：將邊緣從20%增加到40%，確保捕獲完整臉部
5. 調整InsightFace的配置參數：降低檢測閾值，提高靈敏度

使用方法:
-------
python enhanced_onnx_face_dedup.py [參數]

常用參數:
--------
--source: 輸入圖片目錄
--result: 重複臉部圖片輸出目錄
--no-face: 無臉部圖片輸出目錄
--debug: 顯示詳細調試信息
'''

import os
import shutil
import argparse
import time
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import math

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN


# ---------------------------
# Configurable parameters
# ---------------------------
SRC_DIR = "./source"
DST_DIR = "./result"
NO_FACE_DIR = "./no-face"
THRESHOLD = 0.5  # eps for DBSCAN in cosine-distance
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)  # 人臉偵測圖像大小
ONNX_MODEL_PATH = "./webcam-infer-onnx-allan/re_optimized_mbo_bisenetV10_landmark_pose106_fused_model_HWC.onnx"
INSIGHTFACE_DET_THRESH = 0.1  # 降低 InsightFace 的檢測閾值，提高靈敏度
FACE_MARGIN = 0.4  # 臉部邊界擴展比例 (之前是 0.2)
ROTATION_ANGLES = [0, -15, 15, -30, 30]  # 多角度嘗試


# ---------------------------
# CUDA / GPU 檢查
# ---------------------------
def check_cuda_availability(ctx_id: int) -> bool:
    """檢查 CUDA 是否可用，若不可用則回傳 False 以降級到 CPU。"""
    if ctx_id < 0:
        print("CPU 模式已指定 (--ctx-id < 0)")
        return False
    
    # 檢查環境變數是否強制使用 GPU
    force_gpu = os.environ.get('FORCE_GPU', '1').lower() in ('1', 'true', 'yes')
    
    try:
        providers = ort.get_available_providers()
        print(f"可用的 ONNX Runtime 提供者: {providers}")
        has_cuda = 'CUDAExecutionProvider' in providers
        
        if has_cuda:
            if force_gpu:
                print("\n-------------------------------------")
                print("系統檢測到 GPU 支援，並已設置 FORCE_GPU=1")
                print("將使用 GPU 進行運算，可大幅提升速度。")
                print("-------------------------------------\n")
                return True
            else:
                # 原始的 CPU 降級邏輯 (為安全性保留)
                print("\n-------------------------------------")
                print("系統檢測到 GPU 支援，但我們選擇使用 CPU 運行以增加穩定性。")
                print("若您確定 GPU 環境已正確設置，請設置環境變數 FORCE_GPU=1")
                print("或使用 --ctx-id -1 參數明確指定使用 CPU。")
                print("-------------------------------------\n")
                return False
        else:
            print("\n-------------------------------------")
            print("系統未檢測到 CUDA 支援，將使用 CPU 運行。")
            print("如需啟用 GPU 加速，請安裝 NVIDIA 驅動及 CUDA Toolkit。")
            print("並安裝 GPU 版本: pip install onnxruntime-gpu")
            print("-------------------------------------\n")
            return False
    except Exception as e:
        print(f"\n[警告] 檢查 CUDA 時發生錯誤: {e}")
        print("將使用 CPU 模式運行。\n")
        return False


# ---------------------------
# ONNX Model
# ---------------------------
def load_onnx_model(model_path: str, use_gpu: bool):
    """載入 ONNX 模型"""
    print(f"載入 ONNX 模型: {model_path}")
    
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 8
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    try:
        session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
        input_name = session.get_inputs()[0].name
        print(f"ONNX 模型載入成功，輸入名稱: {input_name}")
        return session, input_name
    except Exception as e:
        print(f"載入 ONNX 模型失敗: {e}")
        return None, None


def detect_face_onnx(img_rgb: np.ndarray, ort_session, input_name: str, det_thresh: float = 0.3) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """使用 ONNX 模型偵測人臉和關鍵點"""
    H, W, C = img_rgb.shape
    min_dim = min(H, W)
    scale = 192 / min_dim
    
    # 計算裁剪區域
    offsetx, offsety = 0, 0
    if H > W:
        offsety = (H - W) // 2
    elif W > H:
        offsetx = (W - H) // 2
    
    # 裁剪並縮放到 192x192
    crop_img = img_rgb[offsety:offsety+min_dim, offsetx:offsetx+min_dim]
    resized_img = cv2.resize(crop_img, (192, 192), cv2.INTER_AREA)
    
    # 標準化並添加批次維度
    normalized_img = (resized_img.astype(np.float32) - 127.5) / 127.5
    batch_img = np.expand_dims(normalized_img, 0)
    
    # 執行推論
    try:
        t0 = time.time()
        outputs = ort_session.run([], {input_name: batch_img})
        infer_time = time.time() - t0
        
        # 解析輸出
        landmarks, pose, facemesh, facecorner, cameramatrix = outputs
        
        # 檢查置信度
        confidence = facecorner[0, 4]
        
        if confidence < det_thresh:
            return None
        
        # 將標籤點轉換回原始圖像座標
        landmarks = (landmarks[0] / scale).astype(np.int32)
        if offsetx > 0:
            landmarks[:, 0] += offsetx
        if offsety > 0:
            landmarks[:, 1] += offsety
        
        # 從 facecorner 獲得邊界框 (左上和右下座標)
        bbox = facecorner[0, :4].copy()
        bbox = bbox / scale
        if offsetx > 0:
            bbox[0] += offsetx
            bbox[2] += offsetx
        if offsety > 0:
            bbox[1] += offsety
            bbox[3] += offsety
        
        # 計算完整邊界框 (x1, y1, x2, y2, score 格式) - InsightFace 兼容格式
        x1, y1, x2, y2 = bbox
        final_bbox = np.array([x1, y1, x2, y2, confidence])
        
        return final_bbox, landmarks
    
    except Exception as e:
        print(f"ONNX 推論錯誤: {e}")
        return None


# ---------------------------
# 圖像增強函數
# ---------------------------

def apply_image_enhancements(image: np.ndarray) -> List[np.ndarray]:
    """應用多種圖像增強技術，返回增強後的圖像列表"""
    enhanced_images = []
    
    # 添加原始圖像
    enhanced_images.append(image.copy())
    
    # 亮度調整 (增加亮度)
    brightness_img = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    enhanced_images.append(brightness_img)
    
    # 亮度調整 (降低亮度)
    darkness_img = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
    enhanced_images.append(darkness_img)
    
    # 對比度增強
    contrast_img = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    enhanced_images.append(contrast_img)
    
    # 先灰度化再直方圖均衡化再轉回彩色
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    enhanced_images.append(equalized_rgb)
    
    # 高斯模糊 (輕微)
    blur_img = cv2.GaussianBlur(image, (5, 5), 0)
    enhanced_images.append(blur_img)
    
    # CLAHE (限制對比度自適應直方圖均衡化)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced_images.append(clahe_img)
    
    return enhanced_images


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """旋轉圖像"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 獲取旋轉矩陣
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 計算新圖像的邊界
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # 調整旋轉中心
    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]
    
    # 執行旋轉
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """使用關鍵點對人臉進行對齊"""
    # 為簡單起見，我們以眼睛為關鍵點進行對齊
    # 假設關鍵點0和1代表左右眼
    left_eye = landmarks[36]  # 左眼
    right_eye = landmarks[45]  # 右眼
    
    # 計算眼睛中心點
    left_eye_center = left_eye
    right_eye_center = right_eye
    
    # 計算兩眼之間的角度
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    
    if dx == 0:  # 避免除以零
        angle = 0
    else:
        angle = math.degrees(math.atan2(dy, dx))
    
    # 旋轉圖像
    return rotate_image(image, angle)


# ---------------------------
# 增強特徵提取函數
# ---------------------------

def extract_features_with_enhancements(
    img_rgb: np.ndarray,
    app: FaceAnalysis,
    bbox: np.ndarray,
    landmarks: np.ndarray,
    debug: bool = False
) -> List:
    """使用多種增強技術嘗試提取人臉特徵"""
    # 獲取邊界框
    x1, y1, x2, y2, score = bbox
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img_rgb.shape[1], int(x2)), min(img_rgb.shape[0], int(y2))
    
    # 擴大邊界框，確保包含整個臉部
    w, h = x2 - x1, y2 - y1
    margin_x = int(w * FACE_MARGIN)
    margin_y = int(h * FACE_MARGIN)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img_rgb.shape[1], x2 + margin_x)
    y2 = min(img_rgb.shape[0], y2 + margin_y)
    
    # 裁剪人臉區域
    face_img = img_rgb[y1:y2, x1:x2]
    if face_img.size == 0:
        if debug:
            print("裁剪區域為空")
        return None
    
    # 1. 嘗試提取原始裁剪區域的特徵
    faces = app.get(face_img)
    if faces:
        if debug:
            print("  >> 從原始裁剪區域成功提取特徵")
        return faces
    
    # 2. 嘗試從多個角度提取特徵
    for angle in ROTATION_ANGLES:
        if angle == 0:  # 跳過 0 度，已經處理過
            continue
            
        rotated_img = rotate_image(face_img, angle)
        faces = app.get(rotated_img)
        if faces:
            if debug:
                print(f"  >> 從旋轉角度 {angle}° 成功提取特徵")
            return faces
    
    # 3. 嘗試對齊人臉並提取特徵
    try:
        # 將全局坐標轉換為裁剪區域的局部坐標
        local_landmarks = landmarks.copy()
        local_landmarks[:, 0] = landmarks[:, 0] - x1
        local_landmarks[:, 1] = landmarks[:, 1] - y1
        
        # 過濾掉超出邊界的關鍵點
        valid_points = (local_landmarks[:, 0] >= 0) & (local_landmarks[:, 0] < face_img.shape[1]) & \
                       (local_landmarks[:, 1] >= 0) & (local_landmarks[:, 1] < face_img.shape[0])
        
        if np.any(valid_points):
            aligned_img = align_face(face_img, local_landmarks)
            faces = app.get(aligned_img)
            if faces:
                if debug:
                    print("  >> 從對齊的人臉成功提取特徵")
                return faces
    except Exception as e:
        if debug:
            print(f"  >> 人臉對齊失敗: {e}")
    
    # 4. 嘗試多種圖像增強
    enhanced_images = apply_image_enhancements(face_img)
    for i, enhanced_img in enumerate(enhanced_images):
        faces = app.get(enhanced_img)
        if faces:
            if debug:
                print(f"  >> 從增強圖像 #{i} 成功提取特徵")
            return faces
    
    # 5. 最後嘗試對整張圖像進行處理
    faces = app.get(img_rgb)
    if faces:
        if debug:
            print("  >> 從整張圖像成功提取特徵")
        return faces
    
    # 所有方法都失敗
    if debug:
        print("  >> 所有特徵提取方法都失敗")
    return None


# ---------------------------
# Utilities
# ---------------------------

def is_image_file(fname: str) -> bool:
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="增強型ONNX人臉檢測與去重程序，使用多種圖像增強技術提高特徵提取成功率。")
    parser.add_argument("--source", default=SRC_DIR, help="輸入圖片目錄")
    parser.add_argument("--result", default=DST_DIR, help="重複臉部圖片輸出目錄")
    parser.add_argument("--no-face", default=NO_FACE_DIR, help="無臉部圖片輸出目錄")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="DBSCAN 余弦距離閾值 (較小數值 → 較嚴格)")
    parser.add_argument("--det-thresh", type=float, default=0.3, help="人臉偵測置信度閾值")
    parser.add_argument("--ctx-id", type=int, default=0, help="GPU id; 使用 -1 為只用 CPU")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="InsightFace 模型名稱 (如 buffalo_l, antelopev2)")
    parser.add_argument("--debug", action="store_true", help="顯示除錯訊息")
    parser.add_argument("--det-size", type=int, default=DET_SIZE[0], 
                        help="InsightFace 偵測圖片大小 (較大 = 較精確，較慢)")
    parser.add_argument("--batch-size", type=int, default=16, 
                        help="批次處理圖片數量 (僅適用於 GPU)")
    parser.add_argument("--clear-gpu", action="store_true", 
                        help="每批次後清理 GPU 記憶體")
    parser.add_argument("--onnx-model", default=ONNX_MODEL_PATH,
                        help="ONNX 模型路徑")
    parser.add_argument("--face-margin", type=float, default=FACE_MARGIN,
                        help="人臉邊界擴展比例 (0.4 = 40%)")
    parser.add_argument("--limit", type=int, default=0,
                        help="處理前 N 張圖片 (0 = 處理全部)")
    parser.add_argument("--disable-rotations", action="store_true",
                        help="禁用多角度嘗試")
    parser.add_argument("--disable-alignment", action="store_true",
                        help="禁用人臉對齊")
    parser.add_argument("--disable-enhancements", action="store_true",
                        help="禁用圖像增強")
    return parser.parse_args()


# ---------------------------
# 修復 Windows 下 antelopev2 嵌套文件夾問題
# ---------------------------

def _patch_antelopev2_layout():
    root = Path.home() / '.insightface' / 'models' / 'antelopev2'
    nested = root / 'antelopev2'
    if nested.is_dir():
        root_onnx = list(root.glob('*.onnx'))
        if not root_onnx:
            for onnx_file in nested.glob('*.onnx'):
                target = root / onnx_file.name
                if not target.exists():
                    try:
                        onnx_file.replace(target)
                    except Exception as exc:
                        print(f"[WARN] Failed to move {onnx_file} → {target}: {exc}")
        try:
            nested.rmdir()
        except OSError:
            pass


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    
    # 設置特徵提取增強選項
    global FACE_MARGIN, ROTATION_ANGLES
    FACE_MARGIN = args.face_margin
    if args.disable_rotations:
        ROTATION_ANGLES = [0]
    
    # 先檢查 GPU 可用性
    use_gpu = check_cuda_availability(args.ctx_id)
    if args.debug:
        print(f"[DEBUG] Running with {'GPU' if use_gpu else 'CPU'}")
    
    if not use_gpu and args.ctx_id >= 0:
        print("\n您指定了 GPU 模式 (--ctx-id >= 0)，但 GPU 不可用或未正確設置")
        print("請檢查是否有安裝 CUDA 及 onnxruntime-gpu")
        print("程式將繼續使用 CPU 模式運行，但速度會較慢\n")
        
        # 提示用戶可選擇退出
        if not args.debug:  # debug 模式下不詢問
            response = input("請問要繼續使用 CPU 模式運行嗎? (Y/n): ")
            if response.lower() in ('n', 'no'):
                print("程式結束，請安裝 GPU 依賴後再試")
                return
    
    os.makedirs(args.result, exist_ok=True)
    os.makedirs(args.no_face, exist_ok=True)
    
    _patch_antelopev2_layout()

    providers = ["CPUExecutionProvider"] if not use_gpu else [
        "CUDAExecutionProvider", "CPUExecutionProvider"]
    
    # 準備 InsightFace 模型 (用於特徵提取)
    print(f"載入 InsightFace 模型 {args.model}...")
    app = FaceAnalysis(name=args.model, providers=providers)
    det_size = (args.det_size, args.det_size)
    # 注意：這裡使用較低的閾值以提高靈敏度
    app.prepare(ctx_id=args.ctx_id if use_gpu else -1, 
                det_size=det_size, 
                det_thresh=INSIGHTFACE_DET_THRESH)  # 使用較低的閾值
    print(f"InsightFace 模型載入完成，偵測大小: {det_size}，檢測閾值: {INSIGHTFACE_DET_THRESH}")
    
    # 載入 ONNX 模型
    ort_session = None
    input_name = None
    if os.path.exists(args.onnx_model):
        ort_session, input_name = load_onnx_model(args.onnx_model, use_gpu)
        if ort_session is None:
            print("無法載入 ONNX 模型，程式將退出")
            return
    else:
        print(f"找不到 ONNX 模型: {args.onnx_model}")
        print("請確保 ONNX 模型文件存在")
        return

    # 統計變數
    embeddings: List[np.ndarray] = []
    embed_paths: List[str] = []
    no_face_paths: List[str] = []
    
    # 特徵提取成功統計
    extraction_stats = {
        "total": 0,
        "success": 0,
        "original": 0,
        "rotated": 0,
        "aligned": 0,
        "enhanced": 0,
        "fullimage": 0
    }
    
    # 處理批次大小
    batch_size = args.batch_size if use_gpu else 1  # CPU 不使用批次處理
    
    # 準備所有圖片路徑
    all_images = []
    for fname in sorted(os.listdir(args.source)):
        if not is_image_file(fname):
            continue
        fpath = os.path.join(args.source, fname)
        all_images.append((fpath, fname))
    
    if not all_images:
        print("來源目錄沒有圖片")
        return
    
    # 限制處理的圖片數量
    if args.limit > 0 and args.limit < len(all_images):
        print(f"限制處理前 {args.limit} 張圖片 (共有 {len(all_images)} 張)")
        all_images = all_images[:args.limit]
    
    # 開始處理
    batch_count = 0
    processed_count = 0
    start_time = time.time()
    
    print(f"處理 {len(all_images)} 張圖片，使用增強型特徵提取...")
    
    for fpath, fname in all_images:
        if args.debug:
            print(f"\n處理圖片: {fname}")
        
        img = cv2.imread(fpath)
        if img is None:
            print(f"無法讀取圖片: {fname}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 使用 ONNX 模型偵測人臉
        detect_result = detect_face_onnx(img_rgb, ort_session, input_name, args.det_thresh)
        
        if detect_result is not None:
            bbox, landmarks = detect_result
            
            if args.debug:
                print(f"  >> ONNX 偵測成功! 置信度: {bbox[4]:.3f}")
                
            extraction_stats["total"] += 1
            
            # 使用增強型特徵提取
            faces = extract_features_with_enhancements(
                img_rgb, app, bbox, landmarks, args.debug
            )
            
            if faces:
                extraction_stats["success"] += 1
                if args.debug:
                    print(f"  >> 特徵提取成功! 發現 {len(faces)} 張臉")
                
                # 處理此圖片的臉
                for face in faces:
                    emb = face.embedding.astype(np.float32)
                    # 正規化特徵向量
                    emb /= np.linalg.norm(emb) + 1e-8
                    embeddings.append(emb)
                    embed_paths.append(fpath)
                    if args.debug and len(embeddings) > 1:
                        # 顯示與已有向量的最小距離，方便檢查
                        dmin = min(cosine_distance(emb, e) for e in embeddings[:-1])
                        print(f"  >> {fname}: 最小距離={dmin:.3f}")
            else:
                if args.debug:
                    print(f"  >> 特徵提取失敗")
                no_face_paths.append(fpath)
        else:
            if args.debug:
                print(f"  >> ONNX 偵測失敗")
            no_face_paths.append(fpath)
        
        processed_count += 1
        
        # 批次處理與進度報告
        batch_count += 1
        if batch_count % batch_size == 0 or processed_count == len(all_images):
            elapsed = time.time() - start_time
            avg_time = elapsed / max(1, processed_count)
            remaining = avg_time * (len(all_images) - processed_count)
            
            print(f"已處理 {processed_count}/{len(all_images)} 張圖片，發現 {len(embeddings)} 張臉")
            print(f"時間: {elapsed:.1f}秒, 平均: {avg_time:.3f}秒/張, 預估剩餘: {remaining:.1f}秒")
            
            if extraction_stats["total"] > 0:
                success_rate = extraction_stats["success"] / extraction_stats["total"] * 100
                print(f"特徵提取成功率: {success_rate:.1f}% ({extraction_stats['success']}/{extraction_stats['total']})")
            
            if args.clear_gpu and use_gpu:
                # 清理 GPU 記憶體
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        if args.debug:
                            print("已清理 GPU 記憶體")
                except ImportError:
                    pass
    
    # 處理沒有臉的圖片
    no_face_moved = 0
    if no_face_paths:
        print(f"\n移動 {len(no_face_paths)} 張無臉部圖片到 {args.no_face}...")
        for src in no_face_paths:
            dst = os.path.join(args.no_face, os.path.basename(src))
            # 處理名稱衝突
            if os.path.exists(dst):
                base, ext = os.path.splitext(dst)
                idx = 1
                while os.path.exists(f"{base}_{idx}{ext}"):
                    idx += 1
                dst = f"{base}_{idx}{ext}"
            shutil.move(src, dst)
            no_face_moved += 1
            if args.debug:
                print(f"移動 (無臉): {os.path.basename(src)}")
        print(f"完成移動 {no_face_moved} 張無臉部圖片。")
    
    # 如果沒有足夠的臉，退出
    if len(embeddings) < 2:
        print("偵測到的人臉少於 2 張，無法進行聚類。")
        return
    
    # 使用 DBSCAN 進行聚類
    print(f"對 {len(embeddings)} 張臉進行聚類...")
    embeddings_np = np.vstack(embeddings)
    
    dbs = DBSCAN(metric="cosine", eps=args.threshold, min_samples=1)
    labels = dbs.fit_predict(embeddings_np)
    
    # 將標籤映射到圖片路徑
    from collections import defaultdict
    label2paths = defaultdict(set)
    for path, lb in zip(embed_paths, labels):
        label2paths[lb].add(path)
    
    # 建立移動列表：出現在多張圖片中的臉部
    move_paths = {p for paths in label2paths.values() if len(paths) > 1 for p in paths}
    
    if args.debug:
        print("\n[DEBUG] 聚類摘要:")
        for lb, paths in label2paths.items():
            print(f" 標籤 {lb}: {len(paths)} 張圖片")
    
    if not move_paths:
        print("沒有發現重複臉部，無需移動圖片。")
        return
    
    # 移動包含重複臉部的圖片
    moved = 0
    for src in move_paths:
        dst = os.path.join(args.result, os.path.basename(src))
        # 處理名稱衝突
        if os.path.exists(dst):
            base, ext = os.path.splitext(dst)
            idx = 1
            while os.path.exists(f"{base}_{idx}{ext}"):
                idx += 1
            dst = f"{base}_{idx}{ext}"
        shutil.move(src, dst)
        moved += 1
        if args.debug:
            print(f"移動 (重複臉): {os.path.basename(src)}")
    
    # 總結
    total_time = time.time() - start_time
    print(f"\n完成! 總耗時: {total_time:.1f} 秒")
    print(f"移動了 {moved} 張包含重複臉部的圖片，{no_face_moved} 張無臉部圖片")
    
    # 特徵提取統計
    if extraction_stats["total"] > 0:
        success_rate = extraction_stats["success"] / extraction_stats["total"] * 100
        print(f"\n特徵提取統計:")
        print(f"- 總嘗試次數: {extraction_stats['total']}")
        print(f"- 成功次數: {extraction_stats['success']}")
        print(f"- 成功率: {success_rate:.1f}%")


if __name__ == "__main__":
    main() 