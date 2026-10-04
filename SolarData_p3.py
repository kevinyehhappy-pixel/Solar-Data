from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Tuple
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from SolarData_p2 import SolarDataROI


@dataclass
class SpiculeMeasurement:
    """單一針狀體分析特徵輸出容器 (供模組四導出使用)"""

    spicule_id: int
    xi: float  # 起點 (足點) 像素 X
    yi: float  # 起點 (足點) 像素 Y
    xf: float  # 終點 (尖端) 像素 X
    yf: float  # 終點 (尖端) 像素 Y
    length_pix: float  # 像素長度
    length_km: float  # 實體物理長度 (km)
    length_corrected_km: float  # 投影修正後物理長度 (km)
    theta_raw_deg: float  # 影像坐標系絕對角 (度)
    tilt_to_radial_deg: float  # 相對於日面法線之夾角 (度, 逆時針為正)
    velocity_kms: Optional[float] = None  # 視速度 (km/s, 擬合自 x-t 圖)
    xt_slice: Optional[np.ndarray] = None  # 時空時差矩陣 [Time, Slit_Length]


class SpiculeAnalysisEngine:
    """針狀體特徵工程：影像銳化、長度測量、時空切片與動態運動學分析"""

    def __init__(self, roi: SolarDataROI):
        self.roi = roi
        self.time_cadence = self._extract_time_cadence()
        self.measurements: List[SpiculeMeasurement] = []

    def _extract_time_cadence(self) -> float:
        """解析觀測時間採樣率 (秒/幀)"""
        # 優先從 Primary Header 的 CDELT3 取得
        cadence = self.roi.header.get("CDELT3")
        if cadence is not None and float(cadence) > 0:
            return float(cadence)
        # 若為未知則預設 28.0 秒 (符合多數 IRIS SJI 觀測常規)
        return 28.0

    def enhance_frame(
        self, frame_idx: int = 0, sigma_low: float = 1.0, sigma_high: float = 4.0
    ) -> np.ndarray:
        """利用多尺度帶通高斯差分 (Difference of Gaussians, DoG) 凸顯纖維與針狀體邊緣"""
        raw_img = self.roi.cube[frame_idx].astype(np.float64)
        # 消除微小隨機雜訊
        blur_low = gaussian_filter(raw_img, sigma=sigma_low)
        # 提取大尺度平滑背景
        blur_high = gaussian_filter(raw_img, sigma=sigma_high)
        # 差分提取結構骨架 (反銳化遮罩帶通)
        enhanced = blur_low - blur_high
        return enhanced

    def extract_xt_slice(
        self,
        p_start: Tuple[float, float],
        p_end: Tuple[float, float],
        num_samples: int = 150,
        slit_width: int = 3,
    ) -> np.ndarray:
        """沿著針狀體軸線從所有時間幀中採樣，生成時空時差圖 (x-t Slice)

        :param p_start: (xi, yi) 足點像素坐標
        :param p_end: (xf, yf) 尖端像素坐標
        :param num_samples: 沿狹縫長度採樣的點數
        :param slit_width: 橫向平均寬度 (像素)，提高訊噪比
        """
        n_frames, h, w = self.roi.cube.shape
        xi, yi = p_start
        xf, yf = p_end

        # 生成狹縫中心線採樣路徑
        x_line = np.linspace(xi, xf, num_samples)
        y_line = np.linspace(yi, yf, num_samples)

        # 計算狹縫法線方向 (用於側向寬度積分)
        dx = xf - xi
        dy = yf - yi
        norm = np.hypot(dx, dy)
        if norm == 0:
            raise ValueError("起點與終點不能重合！")

        nx, ny = -dy / norm, dx / norm

        xt_matrix = np.zeros((n_frames, num_samples), dtype=np.float32)

        # 側向寬度採樣偏移量
        offsets = np.arange(
            -(slit_width // 2), slit_width // 2 + 1, dtype=float
        )

        for t in range(n_frames):
            frame = self.roi.cube[t]
            profile_sum = np.zeros(num_samples, dtype=np.float64)

            for off in offsets:
                sample_x = x_line + off * nx
                sample_y = y_line + off * ny
                # 次像素雙三次雙線性插值
                coords = np.vstack((sample_y, sample_x))
                profile_sum += map_coordinates(
                    frame, coords, order=1, mode="nearest"
                )

            xt_matrix[t, :] = profile_sum / len(offsets)

        return xt_matrix

    def analyze_single_spicule(
        self,
        spicule_id: int,
        p_start: Tuple[float, float],
        p_end: Tuple[float, float],
        fit_velocity: bool = True,
        t_start_idx: int = 0,
        t_end_idx: Optional[int] = None,
    ) -> SpiculeMeasurement:
        """對單一針狀體進行幾何定標、相對於日面法線之夾角解算與運動速度擬合"""
        xi, yi = p_start
        xf, yf = p_end

        # 1. 幾何長度計算
        pix_len = float(np.hypot(xf - xi, yf - yi))
        length_km = pix_len * self.roi.scale_km_per_pix

        # 盤面投影縮短補償 (若在邊緣 Limb 則不補償)
        if self.roi.location_type == "disk" and self.roi.foreshortening_mu > 0.1:
            length_corrected_km = length_km / self.roi.foreshortening_mu
        else:
            length_corrected_km = length_km

        # 2. 角度場解算 (相對於日面法線基準)
        theta_raw_rad = np.arctan2(yf - yi, xf - xi)
        theta_raw_deg = float(np.degrees(theta_raw_rad))

        # 相對夾角 = 針狀體角度 - 日面中心指向此處之法線角度
        tilt_to_radial = (
            theta_raw_deg - self.roi.radial_angle_deg + 180.0
        ) % 360.0 - 180.0

        # 3. 時空切片提取與視速度擬合
        xt_slice = self.extract_xt_slice(p_start, p_end)
        velocity_kms = None

        if fit_velocity:
            if t_end_idx is None:
                t_end_idx = min(len(self.roi.cube), t_start_idx + 15)

            # 提取指定時間區間內各時間點的最大強度（脊線位置）
            sub_xt = xt_slice[t_start_idx:t_end_idx, :]
            ridge_positions = np.argmax(sub_xt, axis=1)

            # 時間序列與對應實體距離 (km)
            t_seconds = (
                np.arange(len(ridge_positions)) * self.time_cadence
            )  # 秒
            x_km = (
                (ridge_positions / float(xt_slice.shape[1]))
                * pix_len
                * self.roi.scale_km_per_pix
            )

            # 一階線性回歸擬合斜率: v = dx/dt
            if len(t_seconds) > 2:
                slope, _ = np.polyfit(t_seconds, x_km, 1)
                velocity_kms = float(slope)

        meas = SpiculeMeasurement(
            spicule_id=spicule_id,
            xi=xi,
            yi=yi,
            xf=xf,
            yf=yf,
            length_pix=pix_len,
            length_km=length_km,
            length_corrected_km=length_corrected_km,
            theta_raw_deg=theta_raw_deg,
            tilt_to_radial_deg=tilt_to_radial,
            velocity_kms=velocity_kms,
            xt_slice=xt_slice,
        )
        self.measurements.append(meas)

        print(f"\n[分析結果] 針狀體 #{spicule_id}:")
        print(f"  - 像素長度     : {pix_len:.1f} pix")
        print(
            f"  - 物理長度     : {length_km:.1f} km (投影補償後: {length_corrected_km:.1f} km)"
        )
        print(
            f"  - 法線相對夾角 : {tilt_to_radial:.2f}° (日面法線基準: {self.roi.radial_angle_deg:.2f}°)"
        )
        if velocity_kms is not None:
            print(f"  - 推估噴發視速度: {velocity_kms:.2f} km/s")

        return meas

    def display_spicule_diagnostics(
        self,
        meas: SpiculeMeasurement,
        save_path: str = "./solar_data/spicule_diagnostic.png",
    ):
        """繪製單體針狀體診斷圖 (結構標定圖 + 增強對比圖 + x-t 時空圖)"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. 原始影像疊加起訖點
        frame0 = self.roi.cube[0]
        vmin, vmax = np.percentile(frame0, [1.0, 99.5])
        axes[0].imshow(
            frame0, origin="lower", cmap="gray", vmin=vmin, vmax=vmax
        )
        axes[0].plot(
            [meas.xi, meas.xf],
            [meas.yi, meas.yf],
            color="cyan",
            lw=2,
            marker="o",
            markersize=5,
        )
        axes[0].text(
            meas.xi,
            meas.yi,
            " Start",
            color="lime",
            fontsize=10,
            fontweight="bold",
        )
        axes[0].text(
            meas.xf,
            meas.yf,
            " End",
            color="red",
            fontsize=10,
            fontweight="bold",
        )
        axes[0].set_title(
            f"Spicule #{meas.spicule_id} Vector (L={meas.length_km:.0f} km)"
        )
        axes[0].set_xlabel("X [pix]")
        axes[0].set_ylabel("Y [pix]")

        # 2. DoG 帶通增強骨架圖
        enhanced = self.enhance_frame(0)
        axes[1].imshow(enhanced, origin="lower", cmap="inferno")
        axes[1].plot(
            [meas.xi, meas.xf], [meas.yi, meas.yf], "w--", lw=1.5, alpha=0.8
        )
        axes[1].set_title("Difference-of-Gaussians (DoG) Filter")
        axes[1].set_xlabel("X [pix]")

        # 3. x-t 時空時差切片圖
        if meas.xt_slice is not None:
            total_time_min = (
                meas.xt_slice.shape[0] * self.time_cadence
            ) / 60.0
            im3 = axes[2].imshow(
                meas.xt_slice,
                origin="lower",
                aspect="auto",
                cmap="magma",
                extent=[0, meas.length_km, 0, total_time_min],
            )
            axes[2].set_title(
                f"x-t Space-Time Diagram (v ~ {meas.velocity_kms:.1f} km/s)"
                if meas.velocity_kms
                else "x-t Space-Time Diagram"
            )
            axes[2].set_xlabel("Distance along axis [km]")
            axes[2].set_ylabel("Elapsed Time [minutes]")
            plt.colorbar(im3, ax=axes[2], label="Intensity [DN]")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"  - [儲存完成] 針狀體診斷圖已輸出至: {os.path.abspath(save_path)}")
        plt.show()