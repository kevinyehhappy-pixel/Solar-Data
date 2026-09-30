## 太陽觀測資訊分析

**image input**
- link擷取IRIS網站Mg II& Ha譜線照片
    - 篩選規則:尋找有拍攝到針狀體或
- 擷取所連結照片詳細資料
    - 時間、觀測器材、觀測事件、譜線、可見太陽動態
- 校正區域2D細節
    1. 裁切
    2. 經緯度、平面照片校正
- 邊緣梯度補償
    邊緣因太陽大氣阻擋亮度較低
<hr>

**image analyzing**

使用工具: imagej<br>
主要任務:尋找、分析針狀體<br>
分析:
1. background
    - 磁場分布
        - HMI
        - 位置、具體事件名稱
    - 噴發
        - 都卜勒
        - 位置、具體事件名稱
2. 針狀體
    - 單體
        - 標定起訖點
        - 長度計算
    - 群體
        - 指向性
        - 風花圖繪製
<hr>

**image output**

使用工具: numpy array<br>
輸出內容:

    {Xi, Yi, Xf, Yf, distance, function}