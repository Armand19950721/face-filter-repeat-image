# Face Filter: 人臉去重搬移工具

本專案可自動從資料夾中找出「有重複人臉」的照片，並將這些照片搬移到 result 資料夾，剩下的 source 只保留未重複的人臉。同時，能自動將沒有臉孔的圖片移至 no-face 資料夾。

## 特色
- 使用 InsightFace (SCRFD + ArcFace) 高精度人臉偵測與特徵抽取
- DBSCAN 分群自動判斷同一人
- 支援多臉、多角度、不同光線
- 自動分類無人臉的圖片至獨立資料夾
- 參數可調，適合各種資料集

## 處理流程

1. **人臉偵測**：使用 SCRFD 檢測器找出照片中的所有人臉
   - 使用深度學習模型定位照片中的人臉位置和範圍
   - 為每個人臉保存所在的圖片路徑

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

---

## 使用方式

1. **準備資料夾**
   - 將所有要去重的圖片放到 `./source` 資料夾
   - 確保 `./result` 資料夾存在（或讓程式自動建立）
   - 程式會自動建立 `./no-face` 資料夾存放未偵測到臉孔的圖片

2. **執行主程式**
   ```bash
   python dedup_faces_insight.py
   ```
   預設會：
   - 偵測所有臉
   - 將無人臉的圖片搬到 `./no-face`
   - 自動分群
   - 將有重複人臉的照片搬到 `./result`
   - `./source` 只留下未重複的人臉

3. **常用參數**
   - `--threshold`：分群嚴格度（預設 0.5，數值越小越嚴格）
   - `--det-thresh`：人臉偵測信心分數（預設 0.3，遇到漏檢可調低如 0.15）
   - `--no-face`：指定無人臉圖片的目標資料夾（預設 ./no-face） 
   - `--debug`：顯示每張臉的距離、分群狀況，方便調參

   範例：
   ```bash
   python dedup_faces_insight.py --debug --det-thresh 0.1 --threshold 0.5
   python hybrid_face_dedup.py --debug --det-thresh 0.1 --threshold 0.5 
   ```

---

## 參數說明
| 參數           | 說明                                 | 預設值      |
|----------------|--------------------------------------|-------------|
| --source       | 輸入圖片資料夾                      | ./source    |
| --result       | 搬移重複人臉的資料夾                | ./result    |
| --no-face      | 搬移無人臉圖片的資料夾              | ./no-face   |
| --threshold    | DBSCAN 分群的 cosine 距離閾值        | 0.5         |
| --det-thresh   | 人臉偵測信心分數下限                | 0.3         |
| --model        | InsightFace 模型包（buffalo_l/antelopev2）| buffalo_l |
| --ctx-id       | GPU id（-1 表示只用 CPU）           | 0           |
| --debug        | 顯示 debug 訊息                      | (關閉)      |

---

## 常見問題
- **有些臉沒被偵測到？**
  - 請嘗試 `--det-thresh 0.15` 或調大圖片尺寸
- **同一人不同角度沒被歸為同群？**
  - 請嘗試 `--threshold 0.55` 或 0.6
- **速度慢？**
  - 建議先縮圖到 640px 長邊，或用 GPU 執行

---

## 依賴/引用
- [InsightFace (SCRFD, ArcFace)](https://github.com/deepinsight/insightface)
- [scikit-learn (DBSCAN)](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [OpenCV](https://opencv.org/)

---

## 聯絡/貢獻
如有問題或建議，歡迎開 issue 或 pull request！ 