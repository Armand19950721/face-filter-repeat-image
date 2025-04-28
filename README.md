# Face Filter: 人臉去重搬移工具

本專案可自動從資料夾中找出「有重複人臉」的照片，並將這些照片搬移到 result 資料夾，剩下的 source 只保留未重複的人臉。

## 特色
- 使用 InsightFace (SCRFD + ArcFace) 高精度人臉偵測與特徵抽取
- DBSCAN 分群自動判斷同一人
- 支援多臉、多角度、不同光線
- 參數可調，適合各種資料集

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

2. **執行主程式**
   ```bash
   python dedup_faces_insight.py
   ```
   預設會：
   - 偵測所有臉
   - 自動分群
   - 將有重複人臉的照片搬到 `./result`
   - `./source` 只留下未重複的人臉

3. **常用參數**
   - `--threshold`：分群嚴格度（預設 0.5，數值越小越嚴格）
   - `--det-thresh`：人臉偵測信心分數（預設 0.3，遇到漏檢可調低如 0.15）
   - `--debug`：顯示每張臉的距離、分群狀況，方便調參

   範例：
   ```bash
   python dedup_faces_insight.py --debug --det-thresh 0.07 --threshold 0.5
   ```

---

## 參數說明
| 參數           | 說明                                 | 預設值      |
|----------------|--------------------------------------|-------------|
| --source       | 輸入圖片資料夾                      | ./source    |
| --result       | 搬移重複人臉的資料夾                | ./result    |
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