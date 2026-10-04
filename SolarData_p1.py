import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from sunpy.net import Fido
import astropy.units as u
from sunpy.net import attrs as a


class SolarDataIngestor:

    def __init__(self, download_dir="./solar_data"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.fits_path = None
        self.header = None
        self.data_cube = None

    def search_and_download_iris(
        self, start_time: str, end_time: str, passband: str = "2796"
    ):
        """階段一：檢索並精確下載 IRIS SJI (二維影像) FITS 檔"""
        print(f"\n[階段 1/4] 開始檢索 IRIS 觀測資料庫...")
        print(f"  - 時間區間: {start_time} 至 {end_time}")
        print(f"  - 目標波段: {passband} Å")

        pb = int(passband)
        query = Fido.search(
            a.Time(start_time, end_time),
            a.Instrument("IRIS"),
            a.Level(2),
            a.Wavelength((pb - 2) * u.AA, (pb + 2) * u.AA),
        )

        if query.file_num == 0:
            raise FileNotFoundError(
                "未找到符合條件的 IRIS 觀測數據，請確認時間與波段。"
            )

        client_key = list(query.keys())[0]
        results = query[client_key]
        print(
            f"  - 檢索成功！透過 {client_key.upper()} 共找到 {len(results)} 筆紀錄。"
        )

        # ----------------------------------------------------
        # 關鍵過濾：自動找出包含 SJI 的記錄，避開 Raster
        # ----------------------------------------------------
        target_record = None
        for i, row in enumerate(results):
            # 檢視 URL、檔案名稱或描述欄位是否包含 'sji'
            record_str = str(row).lower()
            if "sji" in record_str:
                target_record = results[i]
                print(f"  - 已自動鎖定 SJI 影像數據 (索引: #{i})")
                break

        # 若都沒標記 sji，預設選取最後一筆（IRIS 查詢中 SJI 通常排在 Raster 後面）
        if target_record is None:
            print("  - [警告] 未明確辨識出 SJI 標籤，嘗試選取末筆記錄。")
            target_record = results[-1]

        # 執行下載
        print(f"[階段 1/4] 下載檔案至本地目錄: {self.download_dir} ...")
        downloaded_files = Fido.fetch(target_record, path=self.download_dir)

        # 支援 .fits 與 .fits.gz
        valid_extensions = (
            ".fits",
            ".fits.gz",
            ".fts",
            ".fts.gz",
            ".fit",
            ".fit.gz",
        )
        fits_files = [
            str(f)
            for f in downloaded_files
            if str(f).lower().endswith(valid_extensions)
        ]

        if not fits_files:
            print(f"\n[偵錯] 下載到的檔案列表: {list(downloaded_files)}")
            raise RuntimeError(
                "下載目標仍非 FITS 檔，請確認該時段是否包含 SJI 觀測。"
            )

        self.fits_path = fits_files[0]
        print(f"  - 取得目標 SJI 檔案: {os.path.basename(self.fits_path)}")
        return self.fits_path

    def load_local_fits(self, fits_path: str):
        """若已有手動下載之 FITS 檔案，可直接指定路徑載入"""
        print(f"\n[階段 1/4] 載入本地 FITS 檔案...")
        if not os.path.exists(fits_path):
            raise FileNotFoundError(f"找不到檔案: {fits_path}")
        self.fits_path = fits_path
        print(f"  - 目標路徑: {self.fits_path}")

    def inspect_and_read_fits(self):
        """階段二：解析 FITS 結構並讀取 Header 與 Data Cube"""
        if not self.fits_path:
            raise ValueError("尚未指定或下載 FITS 檔案。")

        print(f"\n[階段 2/4] 開始讀取與解析 FITS Header 資訊...")

        with fits.open(self.fits_path) as hdul:
            print(f"  - FITS HDU 結構: 共 {len(hdul)} 個 Extension")
            hdul.info()

            # IRIS SJI Level 2 的主要影像立方體通常存放在 PrimaryHDU (Index 0)
            self.header = hdul[0].header
            # 轉換為 float32 以防後續運算記憶體溢位
            self.data_cube = hdul[0].data.astype(np.float32)

        # 擷取關鍵物理參數
        n_frames = self.data_cube.shape[0] if self.data_cube.ndim == 3 else 1
        obs_time = self.header.get("DATE-OBS", "未知")
        telescope = self.header.get("TELESCOP", "未知")
        instrument = self.header.get("INSTRUME", "未知")
        target_wave = self.header.get(
            "TWAVE1", self.header.get("WAVELNTH", "未知")
        )
        xcen = self.header.get("XCEN", "未知")
        ycen = self.header.get("YCEN", "未知")
        cadence = self.header.get("CDELT3", "未知")  # 時間取樣率

        print("\n[階段 3/4] 觀測中繼資料解析完成：")
        print(f"  --------------------------------------------------")
        print(f"  儀器 / 載具     : {telescope} / {instrument}")
        print(f"  起始時間 (UTC)  : {obs_time}")
        print(f"  觀測波長        : {target_wave} Å")
        print(f"  視場中心 (XCEN) : {xcen} arcsec")
        print(f"  視場中心 (YCEN) : {ycen} arcsec")
        print(
            f"  影像維度        : {self.data_cube.shape} (Frames, Y-pixels, X-pixels)"
        )
        print(f"  總幀數 (Frames) : {n_frames}")
        print(f"  時間間隔 (約)   : {cadence} 秒/幀")
        print(f"  --------------------------------------------------")

        return self.header, self.data_cube

    def display_preview(self, frame_idx: int = 0):
        """階段四：視覺化單幀未校正影像，確認日面特徵與結構清晰度"""
        print(
            f"\n[階段 4/4] 正在渲染第 {frame_idx} 幀預覽圖 (Raw Count)..."
        )

        if self.data_cube is None:
            raise ValueError("數據矩陣為空，請先執行 inspect_and_read_fits()")

        frame = (
            self.data_cube[frame_idx]
            if self.data_cube.ndim == 3
            else self.data_cube
        )

        # 濾除宇宙射線造成的異常亮點，取 1% 到 99.5% 亮度拉伸
        vmin, vmax = np.percentile(frame, [1.0, 99.5])

        plt.figure(figsize=(9, 8))
        im = plt.imshow(
            frame,
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Data Number (DN) / Counts", rotation=270, labelpad=15)

        target_wave = self.header.get(
            "TWAVE1", self.header.get("WAVELNTH", "SJI")
        )
        obs_date = self.header.get("DATE-OBS", "")

        plt.title(
            f"IRIS SJI {target_wave} Å Raw Frame: #{frame_idx}\nTime: {obs_date}",
            fontsize=12,
        )
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        plt.grid(color="red", linestyle="--", linewidth=0.5, alpha=0.3)
        plt.tight_layout()

        print(
            f"  - 影像渲染完成 (像素動態範圍限制: {vmin:.1f} ~ {vmax:.1f} DN)"
        )
        plt.show()



if __name__ == "__main__":
    # ----------------------------------------------------
    # 階段 A：執行模組一（讀取本地快取或連網下載）
    # ----------------------------------------------------
    ingestor = SolarDataIngestor(download_dir="./solar_data")

    # 1. 檢索/載入資料 (優先讀取本地已下載的 SJI 檔案)
    ingestor.search_and_download_iris(
        start_time="2022-03-30T17:00:00",
        end_time="2022-03-30T17:10:00",
        passband="2796",
    )

    # 2. 解析 FITS 結構並讀取 3D 數據陣列
    header, raw_cube = ingestor.inspect_and_read_fits()

    # ----------------------------------------------------
    # 階段 B：連動模組二（動態匯入並執行天文定標與裁切）
    # ----------------------------------------------------
    from SolarData_p2 import AstrometryCalibrator

    # 3. 初始化定標器
    calibrator = AstrometryCalibrator(data_cube=raw_cube, header=header)

    # 4. 設定感興趣區域 (ROI) 裁切範圍
    # 方式一：使用像素坐標裁切 (ymin, ymax, xmin, xmax)
    roi_data = calibrator.calibrate_and_crop(pixel_bounds=(100, 300, 80, 280))

    # 方式二：若想改用「日心角秒」裁切，可解除下行註解並註解上方方式一
    # roi_data = calibrator.calibrate_and_crop(arcsec_bounds=(450.0, 550.0, 280.0, 380.0))

    # 5. 顯示經過 WCS 坐標定標、標有「日面法線基準」的科學預覽圖
    calibrator.display_roi_preview(roi_data, frame_idx=0)