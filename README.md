# Face Filter: 人臉去重搬移工具

本專案可自動從資料夾中找出「有重複人臉」的照片，並將這些照片搬移到 result 資料夾，剩下的 source 只保留未重複的人臉。同時，能自動將沒有臉孔的圖片移至 no-face 資料夾。

## 特色
- **多模型檢測**：使用 ONNX、Dlib CNN 和 InsightFace 多種檢測器結合，提高檢測成功率
- **精準人臉偵測**：尤其針對困難角度、光線或被遮擋的人臉有更好表現
- **高質量特徵提取**：使用 InsightFace ArcFace 精確提取臉部特徵
- **智能分群**：DBSCAN 技術自動判斷同一人臉
- **可配置**：多種參數可調整，適應不同應用場景

## 檢測模型比較

| 模型 | 優勢 | 劣勢 | 適用場景 |
|------|------|------|----------|
| ONNX | 速度快、輕量化 | 對非正面人臉效果較差 | 標準正面人像照片 |
| Dlib CNN | 非常精確，支持各種角度、光線 | 速度較慢 | 困難人臉檢測場景 |
| InsightFace | 與特徵提取整合良好 | 特定場景可能失效 | 作為最後備援方案 |

## 處理流程

1. **多模型人臉偵測**：按照設定順序嘗試多種檢測器
   - 預設順序：ONNX → Dlib CNN → InsightFace
   - 一旦成功檢測到人臉就停止嘗試

2. **特徵提取**：使用 ArcFace 為每個人臉生成特徵向量
   - 將每個人臉轉換成512維的特徵向量
   - 標準化向量以便於後續比較

3. **相似度比較**：計算人臉特徵向量間的餘弦距離
   - 距離小於閾值（預設0.5）的人臉被視為相同人
   - 餘弦距離 = 1 - 餘弦相似度

4. **人臉聚類**：使用 DBSCAN 算法將相似人臉分組
   - 自動將相似人臉分到同一群組
   - 不需預先指定人臉數量

5. **照片移動**：
   - 將包含重複人臉的圖片移至 result 目錄
   - 將沒有人臉的圖片移至 no-face 目錄
   - source 目錄保留只有獨特人臉的圖片

---

## 安裝步驟

1. **建立虛擬環境（建議）**
   ```bash
   python -m venv .venv
   # 啟動虛擬環境
   # Windows:
   .\.venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. **安裝相依套件**
   ```bash
   pip install -r requirements.txt
   ```

3. **安裝 Dlib（可選但建議）**
   ```bash
   # 先安裝 CMake (Windows 用戶)
   # 從 https://cmake.org/download/ 下載安裝
   
   # 然後安裝 dlib
   pip install dlib
   ```

---

## 使用方式

1. **準備資料夾**
   - 將所有要去重的圖片放到 `./source` 資料夾
   - 確保 `./result` 資料夾存在（或讓程式自動建立）
   - 程式會自動建立 `./no-face` 資料夾存放未偵測到臉孔的圖片

2. **執行主程式**
   ```bash
   python multi_model_face_demo.py
   ```
   預設會：
   - 按順序使用 ONNX、Dlib CNN、InsightFace 偵測器
   - 將無人臉的圖片搬到 `./no-face`
   - 自動分群
   - 將有重複人臉的照片搬到 `./result`
   - `./source` 只留下未重複的人臉

3. **進階用法**
   ```bash
   # 使用純 InsightFace 檢測，適合速度優先的情況
   python multi_model_face_demo.py --detector-order insightface
   
   # 使用 Dlib CNN 高精度檢測，提高上採樣次數以檢測小臉
   python multi_model_face_demo.py --detector-order dlib --dlib-upsample 2
   
   # 完整的檢測管道，對於困難圖像，提高置信度閾值
   python multi_model_face_demo.py --det-thresh 0.1
   ```

4. **常用參數**
   - `--threshold`：分群嚴格度（預設 0.5，數值越小越嚴格）
   - `--det-thresh`：人臉偵測信心分數（預設 0.1，遇到漏檢可調低）
   - `--no-face`：指定無人臉圖片的目標資料夾（預設 ./no-face） 
   - `--debug`：顯示每張臉的距離、分群狀況，方便調參
   - `--detector-order`：檢測器使用順序，如：`onnx,dlib,insightface`
   - `--dlib-upsample`：Dlib CNN 上採樣次數 (0-2)，提高可檢測更小的臉
   - `--skip-dlib`：跳過 Dlib CNN 檢測以提高速度

---

## 參數說明
| 參數           | 說明                                 | 預設值      |
|----------------|--------------------------------------|-------------|
| --source       | 輸入圖片資料夾                      | ./source    |
| --result       | 搬移重複人臉的資料夾                | ./result    |
| --no-face      | 搬移無人臉圖片的資料夾              | ./no-face   |
| --threshold    | DBSCAN 分群的 cosine 距離閾值        | 0.5         |
| --det-thresh   | 人臉偵測信心分數下限                | 0.1         |
| --model        | InsightFace 模型包（buffalo_l/antelopev2）| buffalo_l |
| --ctx-id       | GPU id（-1 表示只用 CPU）           | 0           |
| --detector-order | 檢測器使用順序 (逗號分隔)           | onnx,dlib,insightface |
| --dlib-upsample | Dlib CNN 上採樣次數 (0-2)           | 1           |
| --debug        | 顯示 debug 訊息                      | (關閉)      |

---

## 常見問題
- **有些臉沒被偵測到？**
  - 嘗試 `--det-thresh 0.05` 降低置信度閾值
  - 增加 `--dlib-upsample 2` 提高 Dlib 檢測小臉的能力
  - 調整檢測器順序 `--detector-order dlib,onnx,insightface` 優先使用更準確的檢測器
- **同一人不同角度沒被歸為同群？**
  - 請嘗試 `--threshold 0.55` 或 0.6
- **運行速度慢？**
  - 使用 `--skip-dlib` 跳過 Dlib CNN 檢測器（會犧牲一些準確度）
  - 使用 `--detector-order onnx,insightface` 只用較快的檢測器
  - 建議用 GPU 執行以獲得更好性能

---

## 依賴/引用
- [InsightFace (SCRFD, ArcFace)](https://github.com/deepinsight/insightface)
- [Dlib (CNN Face Detector)](http://dlib.net/)
- [scikit-learn (DBSCAN)](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [OpenCV](https://opencv.org/)

---

## 聯絡/貢獻
如有問題或建議，歡迎開 issue 或 pull request！ 