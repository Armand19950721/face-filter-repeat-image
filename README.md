# 增強型ONNX人臉去重程序 (Enhanced ONNX Face Deduplication)

該程序是一個基於ONNX模型的人臉檢測與特徵提取系統，專門用於處理難以識別的人臉圖像。此版本專注於提高特徵提取成功率，實現了多種圖像增強技術。

## 特徵提取增強技術

1. **多角度嘗試**：在特徵提取前對裁剪區域進行輕微旋轉（±15°，±30°）
2. **多種圖像增強組合**：亮度調整、對比度增強、高斯模糊、直方圖均衡化等
3. **人臉對齊**：使用檢測到的關鍵點進行面部對齊，再提取特徵
4. **擴大邊界**：將邊緣從20%增加到40%，確保捕獲完整臉部
5. **調整InsightFace的配置參數**：降低檢測閾值，提高靈敏度

## 環境要求

- Python 3.8+
- CUDA支持（GPU加速）
- 依賴庫：
  - onnxruntime-gpu (GPU運行)
  - insightface
  - opencv-python
  - numpy
  - scikit-learn

## 安裝步驟

1. 克隆或下載本倉庫
2. 安裝依賴：
```
pip install -r requirements.txt
```
3. 安裝GPU支持（可選但推薦）：
```
pip install onnxruntime-gpu
```

## 使用方法

基本用法：

```
python enhanced_onnx_face_dedup.py --source ./source --result ./result --no-face ./no-face
```

### 命令行參數

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--source` | 輸入圖片目錄 | ./source |
| `--result` | 重複臉部圖片輸出目錄 | ./result |
| `--no-face` | 無臉部圖片輸出目錄 | ./no-face |
| `--threshold` | DBSCAN 余弦距離閾值 (較小數值 → 較嚴格) | 0.5 |
| `--det-thresh` | 人臉偵測置信度閾值 | 0.3 |
| `--ctx-id` | GPU id; 使用 -1 為只用 CPU | 0 |
| `--model` | InsightFace 模型名稱 | buffalo_l |
| `--debug` | 顯示除錯訊息 | False |
| `--det-size` | InsightFace 偵測圖片大小 | 640 |
| `--batch-size` | 批次處理圖片數量 (僅適用於 GPU) | 16 |
| `--clear-gpu` | 每批次後清理 GPU 記憶體 | False |
| `--face-margin` | 人臉邊界擴展比例 | 0.4 |
| `--limit` | 處理前 N 張圖片 (0 = 處理全部) | 0 |
| `--disable-rotations` | 禁用多角度嘗試 | False |
| `--disable-alignment` | 禁用人臉對齊 | False |
| `--disable-enhancements` | 禁用圖像增強 | False |

## GPU支持設置

本程序支持GPU加速，推薦使用NVIDIA GPU運行。配置GPU支持的方法：

1. 確保已安裝CUDA和cuDNN
2. 安裝GPU版本的ONNX Runtime: `pip install onnxruntime-gpu`
3. 使用以下方式啟用GPU:
   - 設置環境變數: `set FORCE_GPU=1` (Windows) 或 `export FORCE_GPU=1` (Linux/Mac)
   - 或修改程式碼中的默認設置（已設為自動使用GPU）

### 為RTX顯卡優化的配置

如果您使用RTX系列顯卡(如RTX 3070)，建議使用以下參數以獲得最佳性能：

```
python enhanced_onnx_face_dedup.py --batch-size 32 --det-size 960
```

## 程序運行流程

1. 載入人臉檢測和特徵提取模型
2. 掃描源目錄中的所有圖像
3. 對每張圖像進行人臉檢測
4. 使用增強技術對難識別的人臉進行處理
5. 提取人臉特徵向量
6. 使用DBSCAN聚類算法對相似人臉進行分組
7. 將包含重複人臉的圖像移至結果目錄
8. 將無人臉的圖像移至指定目錄

## 故障排除

- **GPU不可用**: 如果出現`LoadLibrary failed with error 126`等錯誤，表示CUDA庫無法載入
  - 解決方案: 重新安裝CUDA Toolkit和onnxruntime-gpu，確保版本兼容
  - 示例: `pip install onnxruntime-gpu==1.16.3`
- **記憶體不足**: 降低批次大小(`--batch-size`)和檢測尺寸(`--det-size`)
- **人臉檢測失敗**: 使用`--debug`模式檢查詳細日誌，嘗試調整檢測閾值(`--det-thresh`)

---

## 依賴/引用
- [InsightFace (SCRFD, ArcFace)](https://github.com/deepinsight/insightface)
- [Dlib (CNN Face Detector)](http://dlib.net/)
- [scikit-learn (DBSCAN)](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [OpenCV](https://opencv.org/)

---

## 聯絡/貢獻
如有問題或建議，歡迎開 issue 或 pull request！ 