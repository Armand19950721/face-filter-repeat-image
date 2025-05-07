import os
import shutil
import argparse
from typing import List, Tuple, Optional
from pathlib import Path
import warnings
import time

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
NO_FACE_DIR = "./no-face"  # Default directory for images with no faces
THRESHOLD = 0.5  # eps for DBSCAN in cosine-distance
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)  # 人臉偵測圖像大小
ONNX_MODEL_PATH = "./webcam-infer-onnx-allan/re_optimized_mbo_bisenetV10_landmark_pose106_fused_model_HWC.onnx"


# ---------------------------
# CUDA / GPU 檢查
# ---------------------------
def check_cuda_availability(ctx_id: int) -> bool:
    """檢查 CUDA 是否可用，若不可用則回傳 False 以降級到 CPU。"""
    if ctx_id < 0:
        print("CPU 模式已指定 (--ctx-id < 0)")
        return False
    
    # 檢查環境變數是否強制使用 GPU
    force_gpu = os.environ.get('FORCE_GPU', '0').lower() in ('1', 'true', 'yes')
    
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
# Allan's ONNX Model
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
# Utilities
# ---------------------------

def is_image_file(fname: str) -> bool:
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move all images that contain faces appearing more than once into ./result using hybrid face detection.")
    parser.add_argument("--source", default=SRC_DIR, help="Directory with input images")
    parser.add_argument("--result", default=DST_DIR, help="Directory to move duplicated-face images")
    parser.add_argument("--no-face", default=NO_FACE_DIR, help="Directory to move images with no detected faces")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Cosine-distance eps used in DBSCAN (smaller → stricter)")
    parser.add_argument("--det-thresh", type=float, default=0.3, help="Face detection confidence threshold")
    parser.add_argument("--ctx-id", type=int, default=0, help="GPU id; use -1 for CPU only")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="InsightFace model pack name (e.g. buffalo_l, antelopev2)")
    parser.add_argument("--debug", action="store_true", help="Print extra debugging info")
    parser.add_argument("--det-size", type=int, default=DET_SIZE[0], 
                        help="Size of detection image (larger = more accurate, slower)")
    parser.add_argument("--batch-size", type=int, default=16, 
                        help="Process images in batches if >1 (GPU only)")
    parser.add_argument("--clear-gpu", action="store_true", 
                        help="Clear GPU memory after feature extraction (helps with memory limits)")
    parser.add_argument("--onnx-model", default=ONNX_MODEL_PATH,
                        help="Path to Allan's ONNX face detection model")
    parser.add_argument("--use-insightface-det", action="store_true",
                        help="Use InsightFace for detection instead of ONNX model")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit processing to the first N images (0 = process all)")
    # 保留這個參數定義，但我們將禁用其功能
    parser.add_argument("--cascade", action="store_true",
                        help="Use cascade detection: try InsightFace if ONNX detection fails")
    return parser.parse_args()


# ---------------------------
# Helper to fix Windows nested folder issue
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
    os.makedirs(args.no_face, exist_ok=True)  # Create no-face directory
    
    _patch_antelopev2_layout()

    providers = ["CPUExecutionProvider"] if not use_gpu else [
        "CUDAExecutionProvider", "CPUExecutionProvider"]
    
    # 準備 InsightFace 模型 (僅用於特徵提取)
    print(f"Loading InsightFace model {args.model}...")
    app = FaceAnalysis(name=args.model, providers=providers)
    det_size = (args.det_size, args.det_size)
    app.prepare(ctx_id=args.ctx_id if use_gpu else -1, 
                det_size=det_size, 
                det_thresh=args.det_thresh)
    print(f"InsightFace model loaded with detection size {det_size}")
    
    # 強制使用 ONNX 模型
    args.use_insightface_det = False
    
    # 載入 Allan 的 ONNX 模型
    ort_session = None
    input_name = None
    if os.path.exists(args.onnx_model):
        print(f"Loading Allan's ONNX model: {args.onnx_model}")
        ort_session, input_name = load_onnx_model(args.onnx_model, use_gpu)
        if ort_session is None:
            print("無法載入 ONNX 模型，程式將退出")
            return
    else:
        print(f"找不到 ONNX 模型: {args.onnx_model}")
        print("請確保 ONNX 模型文件存在")
        return

    embeddings: List[np.ndarray] = []
    embed_paths: List[str] = []  # one path per embedding
    no_face_paths: List[str] = []  # Track paths of images with no faces
    batch_size = args.batch_size if use_gpu else 1  # CPU 不使用批次處理

    # 準備所有圖片路徑
    all_images = []
    for fname in sorted(os.listdir(args.source)):
        if not is_image_file(fname):
            continue
        fpath = os.path.join(args.source, fname)
        all_images.append((fpath, fname))

    if not all_images:
        print("No images found in source directory.")
        return
    
    # 限制處理的圖片數量
    if args.limit > 0 and args.limit < len(all_images):
        print(f"Limiting to first {args.limit} images (out of {len(all_images)} total)")
        all_images = all_images[:args.limit]

    # ---------- Extract embeddings ----------
    batch_count = 0
    processed_count = 0
    print(f"Processing {len(all_images)} images...")
    
    start_time = time.time()
    onnx_success = 0
    
    for fpath, fname in all_images:
        img = cv2.imread(fpath)
        if img is None:
            if args.debug:
                print(f"[DEBUG] Could not read {fname}")
            continue
        
        # 處理圖片
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 使用 Allan 的 ONNX 模型進行人臉偵測
        faces = None
        
        # 使用 Allan 的 ONNX 模型偵測人臉
        detect_result = detect_face_onnx(img_rgb, ort_session, input_name, args.det_thresh)
        
        if detect_result is not None:
            # 從 ONNX 模型獲取臉部位置
            bbox, landmarks = detect_result
            onnx_success += 1
            
            # 剪裁人臉區域
            x1, y1, x2, y2, score = bbox
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img_rgb.shape[1], int(x2)), min(img_rgb.shape[0], int(y2))
            
            # 擴大邊界框確保包含整個臉
            margin = int(max(x2-x1, y2-y1) * 0.2)  # 20% 的邊緣
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img_rgb.shape[1], x2 + margin)
            y2 = min(img_rgb.shape[0], y2 + margin)
            
            # 將臉部區域傳給 InsightFace 進行特徵向量提取
            faces = app.get(img_rgb)
            
            # 如果 InsightFace 沒有檢測到人臉，但 ONNX 檢測到了
            # 我們嘗試將裁剪的人臉圖像傳給 InsightFace
            if not faces:
                face_img = img_rgb[y1:y2, x1:x2]
                if face_img.size > 0:  # 確保有效的裁剪
                    faces = app.get(face_img)
                    if faces and args.debug:
                        print(f"[DEBUG] InsightFace detected face in ONNX crop: {fname}")
        
        # 以下是級聯檢測代碼，已被註釋掉
        # elif args.cascade and not faces:
        #     if args.debug:
        #         print(f"[DEBUG] ONNX failed, trying InsightFace: {fname}")
        #     faces = app.get(img_rgb)
        #     if faces:
        #         cascade_success += 1
        #         if args.debug:
        #             print(f"[DEBUG] InsightFace successful in cascade: {fname}")
        
        processed_count += 1
        
        if not faces:
            if args.debug:
                print(f"[DEBUG] No face: {fname}")
            no_face_paths.append(fpath)  # Add to no-face list
            continue
        
        # 處理此圖片的臉
        for face in faces:
            emb = face.embedding.astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-8
            embeddings.append(emb)
            embed_paths.append(fpath)
            if args.debug and len(embeddings) > 1:
                # print min distance to previous embeddings for quick inspection
                dmin = min(cosine_distance(emb, e) for e in embeddings[:-1])
                print(f"[DEBUG] {fname:15s} minDist={dmin:.3f}")
        
        # 批次處理計數與進度報告
        batch_count += 1
        if batch_count % batch_size == 0 or processed_count == len(all_images):
            elapsed = time.time() - start_time
            avg_time = elapsed / max(1, processed_count)
            remaining = avg_time * (len(all_images) - processed_count)
            
            print(f"Processed {processed_count}/{len(all_images)} images, found {len(embeddings)} faces")
            print(f"Time: {elapsed:.1f}s, Avg: {avg_time:.3f}s/img, Est. remaining: {remaining:.1f}s")
            
            # 打印檢測成功統計
            total_detected = processed_count - len(no_face_paths)
            if total_detected > 0:
                print(f"Detection success: ONNX: {onnx_success} ({onnx_success/total_detected*100:.1f}%)")
            
            if args.clear_gpu and use_gpu:
                # 清理 GPU 記憶體 (可選)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    # 處理沒有臉的圖片
    no_face_moved = 0
    if no_face_paths:
        print(f"\nMoving {len(no_face_paths)} images with no faces to {args.no_face}...")
        for src in no_face_paths:
            dst = os.path.join(args.no_face, os.path.basename(src))
            # handle name clash
            if os.path.exists(dst):
                base, ext = os.path.splitext(dst)
                idx = 1
                while os.path.exists(f"{base}_{idx}{ext}"):
                    idx += 1
                dst = f"{base}_{idx}{ext}"
            shutil.move(src, dst)
            no_face_moved += 1
            print(f"MOVE (no face): {os.path.basename(src)}")
        print(f"Finished moving {no_face_moved} images with no faces.")

    # 錯誤處理：若無臉或僅一張臉則提前結束
    if len(embeddings) < 2:
        print("Less than 2 faces detected. Nothing to cluster.")
        return

    print(f"Clustering {len(embeddings)} faces...")
    embeddings_np = np.vstack(embeddings)

    # ---------- Cluster with DBSCAN ----------
    dbs = DBSCAN(metric="cosine", eps=args.threshold, min_samples=1)
    labels = dbs.fit_predict(embeddings_np)

    # Map label → unique image paths
    from collections import defaultdict
    label2paths = defaultdict(set)
    for path, lb in zip(embed_paths, labels):
        label2paths[lb].add(path)

    # Build move list: clusters appearing in more than one image
    move_paths = {p for paths in label2paths.values() if len(paths) > 1 for p in paths}

    if args.debug:
        print("\n[DEBUG] Cluster summary:")
        for lb, paths in label2paths.items():
            print(f" label {lb}: {len(paths)} images")

    if not move_paths:
        print("No duplicated faces found. Nothing to move.")
        return

    moved = 0
    for src in move_paths:
        dst = os.path.join(args.result, os.path.basename(src))
        # handle name clash
        if os.path.exists(dst):
            base, ext = os.path.splitext(dst)
            idx = 1
            while os.path.exists(f"{base}_{idx}{ext}"):
                idx += 1
            dst = f"{base}_{idx}{ext}"
        shutil.move(src, dst)
        moved += 1
        print(f"MOVE : {os.path.basename(src)}")

    # 計算總處理時間
    total_time = time.time() - start_time
    print(f"\nFinished in {total_time:.1f} seconds.")
    print(f"Moved {moved} images containing duplicated faces and {no_face_moved} images with no faces.")
    
    # 打印檢測統計
    total_detected = processed_count - no_face_moved
    if total_detected > 0:
        print(f"\nDetection statistics:")
        print(f"- ONNX success: {onnx_success} ({onnx_success/total_detected*100:.1f}%)")
        # 級聯檢測統計已移除
        print(f"- Total detection rate: {onnx_success/processed_count*100:.1f}%")
        print(f"- No face rate: {no_face_moved/processed_count*100:.1f}%")


if __name__ == "__main__":
    main() 