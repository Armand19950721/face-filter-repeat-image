import os
import shutil
import argparse
from typing import List
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN  # NEW


# ---------------------------
# Configurable parameters
# ---------------------------
SRC_DIR = "./source"
DST_DIR = "./result"
THRESHOLD = 0.5  # eps for DBSCAN in cosine-distance
MODEL_NAME = "buffalo_l"


# ---------------------------
# Utilities
# ---------------------------

def is_image_file(fname: str) -> bool:
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move all images that contain faces appearing more than once into ./result using InsightFace embeddings + DBSCAN clustering.")
    parser.add_argument("--source", default=SRC_DIR, help="Directory with input images")
    parser.add_argument("--result", default=DST_DIR, help="Directory to move duplicated-face images")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Cosine-distance eps used in DBSCAN (smaller → stricter)")
    parser.add_argument("--det-thresh", type=float, default=0.3, help="Face detection confidence threshold")
    parser.add_argument("--ctx-id", type=int, default=0, help="GPU id; use -1 for CPU only")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="InsightFace model pack name (e.g. buffalo_l, antelopev2)")
    parser.add_argument("--debug", action="store_true", help="Print extra debugging info")
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
    os.makedirs(args.result, exist_ok=True)

    _patch_antelopev2_layout()

    providers = ["CPUExecutionProvider"] if args.ctx_id < 0 else [
        "CUDAExecutionProvider", "CPUExecutionProvider"]
    print("Loading InsightFace model …")
    app = FaceAnalysis(name=args.model, providers=providers)
    app.prepare(ctx_id=args.ctx_id, det_size=(640, 640), det_thresh=args.det_thresh)

    embeddings: List[np.ndarray] = []
    embed_paths: List[str] = []  # one path per embedding

    # ---------- Extract embeddings ----------
    for fname in sorted(os.listdir(args.source)):
        if not is_image_file(fname):
            continue
        fpath = os.path.join(args.source, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not faces:
            if args.debug:
                print(f"[DEBUG] No face: {fname}")
            continue
        for face in faces:
            emb = face.embedding.astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-8
            embeddings.append(emb)
            embed_paths.append(fpath)
            if args.debug and len(embeddings) > 1:
                # print min distance to previous embeddings for quick inspection
                dmin = min(cosine_distance(emb, e) for e in embeddings[:-1])
                print(f"[DEBUG] {fname:15s} minDist={dmin:.3f}")

    if len(embeddings) < 2:
        print("Less than 2 faces detected. Nothing to do.")
        return

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

    print(f"\nFinished. Moved {moved} images containing duplicated faces.")


if __name__ == "__main__":
    main() 