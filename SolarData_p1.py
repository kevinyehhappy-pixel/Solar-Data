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
    print(
        "======================================================================"
    )
    print(" 太陽針狀體動態觀測分析管線 (IRIS SJI 2796 Å Pipeline) 啟動")
    print(
        "======================================================================"
    )

    # ----------------------------------------------------
    # [模組一] 數據檢索與 FITS 讀取
    # ----------------------------------------------------
    ingestor = SolarDataIngestor(download_dir="./solar_data")
    local_sji_file = "./solar_data/iris_l2_20220330_161411_3660259102_SJI_2796_t000.fits.gz"

    if os.path.exists(local_sji_file):
        ingestor.load_local_fits(local_sji_file)
    else:
        ingestor.search_and_download_iris(
            start_time="2022-03-30T17:00:00",
            end_time="2022-03-30T17:10:00",
            passband="2796",
        )

    header, raw_cube = ingestor.inspect_and_read_fits()

    # ----------------------------------------------------
    # [模組二] 天文物理定標與 ROI 裁切
    # ----------------------------------------------------
    from SolarData_p2 import AstrometryCalibrator

    calibrator = AstrometryCalibrator(data_cube=raw_cube, header=header)
    # 裁切 200x200 像素之目標區域
    roi_data = calibrator.calibrate_and_crop(pixel_bounds=(100, 300, 80, 280))

    # ----------------------------------------------------
    # [模組三] 針狀體幾何標定、時空切片與運動學擬合
    # ----------------------------------------------------
    from SolarData_p3 import SpiculeAnalysisEngine

    engine = SpiculeAnalysisEngine(roi=roi_data)

    # 模擬採樣一組針狀體簇 (足點 -> 尖端像素坐標)
    test_spicules = [
        (1, (45, 50), (60, 95)),
        (2, (80, 70), (105, 120)),
        (3, (120, 110), (145, 150)),
        (4, (70, 140), (85, 175)),
        (5, (130, 40), (160, 80)),
    ]

    print(
        f"\n[階段 3/4] 開始批量分析 {len(test_spicules)} 條針狀體特徵與視速度..."
    )
    for sp_id, p_start, p_end in test_spicules:
        engine.analyze_single_spicule(
            spicule_id=sp_id,
            p_start=p_start,
            p_end=p_end,
            fit_velocity=True,
            t_start_idx=0,
            t_end_idx=15,
        )

    # 輸出單體診斷圖 (以第 1 條為例)
    engine.display_spicule_diagnostics(
        engine.measurements[0], save_path="./solar_data/spicule_diagnostic.png"
    )

    # ----------------------------------------------------
    # [模組四] 結構化導出、極坐標風花圖與向量圖譜
    # ----------------------------------------------------
    from SolarData_p4 import SolarDataExporter

    exporter = SolarDataExporter(
        roi=roi_data, measurements=engine.measurements
    )

    # 導出標準表格與二進位陣列
    csv_path = exporter.export_to_csv("spicule_catalog.csv")
    npy_path = exporter.export_to_numpy("spicule_catalog.npy")

    # 繪製群體指向風花圖 (Polar Wind-Rose)
    exporter.plot_wind_rose(
        num_bins=16, save_name="wind_rose_spicules.png", show=False
    )

    # 繪製全視場疊加向量圖譜
    exporter.plot_overlay_catalog(
        frame_idx=0, save_name="spicules_overlay.png", show=False
    )

    print(
        "\n======================================================================"
    )
    print(" 全分析流程執行完成！產出檔案清單：")
    print(f"  1. 幾何定標預覽圖 : ./solar_data/roi_preview.png")
    print(f"  2. 單體診斷時空圖 : ./solar_data/spicule_diagnostic.png")
    print(f"  3. 群體極坐標風花圖: ./solar_data/wind_rose_spicules.png")
    print(f"  4. 視場向量疊加圖 : ./solar_data/spicules_overlay.png")
    print(f"  5. 結構化特徵目錄 : {csv_path}")
    print(
        "======================================================================"
    )