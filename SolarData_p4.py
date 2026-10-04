import csv
from dataclasses import asdict
import os
from typing import List, Optional
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from SolarData_p2 import SolarDataROI
from SolarData_p3 import SpiculeMeasurement


class SolarDataExporter:
    """模組四：結構化科學數據導出、極坐標風花圖與多特徵疊加可視化"""

    def __init__(
        self,
        roi: SolarDataROI,
        measurements: List[SpiculeMeasurement],
        output_dir: str = "./solar_data",
    ):
        self.roi = roi
        self.measurements = measurements
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.measurements:
            print("[警告] 尚未輸入任何針狀體測量數據！")

    def export_to_csv(self, filename: str = "spicule_catalog.csv") -> str:
        """導出標準 CSV 表格檔案"""
        file_path = os.path.join(self.output_dir, filename)
        fieldnames = [
            "spicule_id",
            "xi",
            "yi",
            "xf",
            "yf",
            "length_pix",
            "length_km",
            "length_corrected_km",
            "theta_raw_deg",
            "tilt_to_radial_deg",
            "velocity_kms",
        ]

        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            for m in self.measurements:
                writer.writerow(asdict(m))

        print(
            f"[導出成功] 針狀體特徵目錄 (CSV): {os.path.abspath(file_path)} (共 {len(self.measurements)} 筆)"
        )
        return file_path

    def export_to_numpy(self, filename: str = "spicule_catalog.npy") -> str:
        """導出結構化 NumPy 矩陣 (保存高速二進位數值)"""
        file_path = os.path.join(self.output_dir, filename)

        dtype = [
            ("id", "i4"),
            ("xi", "f8"),
            ("yi", "f8"),
            ("xf", "f8"),
            ("yf", "f8"),
            ("length_km", "f8"),
            ("tilt_radial_deg", "f8"),
            ("velocity_kms", "f8"),
        ]

        data_rows = []
        for m in self.measurements:
            vel = m.velocity_kms if m.velocity_kms is not None else np.nan
            data_rows.append(
                (
                    m.spicule_id,
                    m.xi,
                    m.yi,
                    m.xf,
                    m.yf,
                    m.length_corrected_km,
                    m.tilt_to_radial_deg,
                    vel,
                )
            )

        structured_arr = np.array(data_rows, dtype=dtype)
        np.save(file_path, structured_arr)
        print(
            f"[導出成功] 結構化二進位數值 (NumPy): {os.path.abspath(file_path)}"
        )
        return file_path

    def plot_wind_rose(
        self,
        num_bins: int = 18,
        save_name: str = "spicule_wind_rose.png",
        show: bool = True,
    ):
        """繪製以「局部日面法線為正上方 0°」的極坐標風花分佈圖 (Polar Wind-Rose)"""
        if not self.measurements:
            return

        # 提取相對於法線的夾角 (度)
        tilts_deg = np.array(
            [m.tilt_to_radial_deg for m in self.measurements]
        )
        # 轉換為弧度，範圍限縮於 [-pi, pi]
        tilts_rad = np.radians(tilts_deg)

        # 統計直方圖區間
        bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        counts, bin_edges = np.histogram(tilts_rad, bins=bins)
        widths = np.diff(bin_edges)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")

        # 太陽物理規範：正上方（North / 90°位置）設為 0° 法線基準
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # 順時針為正偏角

        # 柱狀圖色彩以長度或數量映射
        colors = cm.plasma(counts / (max(counts) if max(counts) > 0 else 1.0))
        bars = ax.bar(
            centers,
            counts,
            width=widths,
            bottom=0.0,
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )

        # 設定角度標籤 (-90° ~ +90°)
        ax.set_thetagrids(
            np.arange(0, 360, 45),
            labels=[
                "0° (Radial)",
                "+45°",
                "+90°",
                "+135°",
                "180°",
                "-135°",
                "-90°",
                "-45°",
            ],
        )

        mean_tilt = np.mean(tilts_deg)
        std_tilt = np.std(tilts_deg)
        ax.set_title(
            f"Spicule Orientation Wind-Rose\n"
            f"Reference: Local Solar Radial Vector\n"
            f"Sample: N={len(tilts_deg)} | Mean: {mean_tilt:.1f}° ± {std_tilt:.1f}°",
            va="bottom",
            fontsize=11,
            pad=15,
        )

        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"[繪圖完成] 風花圖已輸出至: {os.path.abspath(save_path)}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_overlay_catalog(
        self,
        frame_idx: int = 0,
        save_name: str = "spicule_overlay_map.png",
        show: bool = True,
    ):
        """在 ROI 影像上繪製所有偵測到的針狀體向量軌跡與編號"""
        if not self.measurements:
            return

        frame = self.roi.cube[frame_idx]
        vmin, vmax = np.percentile(frame, [1.0, 99.5])

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection=self.roi.wcs)

        im = ax.imshow(
            frame,
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensity [DN]", rotation=270, labelpad=15)

        # 疊加每條針狀體向量箭頭
        for m in self.measurements:
            dx = m.xf - m.xi
            dy = m.yf - m.yi
            ax.annotate(
                "",
                xy=(m.xf, m.yf),
                xytext=(m.xi, m.yi),
                arrowprops=dict(
                    arrowstyle="->", color="cyan", lw=1.5, mutation_scale=15
                ),
            )
            ax.plot(m.xi, m.yi, marker="o", color="lime", markersize=3)
            # 標註 ID 與速度
            v_str = (
                f"{m.velocity_kms:.1f} km/s"
                if m.velocity_kms is not None
                else ""
            )
            ax.text(
                m.xf + 1,
                m.yf + 1,
                f"#{m.spicule_id} {v_str}",
                color="yellow",
                fontsize=8,
                weight="bold",
            )

        ax.set_title(
            f"IRIS SJI 2796 Å - Detected Spicules Overlay (N={len(self.measurements)})\n"
            f"Scale: {self.roi.scale_km_per_pix:.2f} km/pix",
            fontsize=12,
        )
        ax.set_xlabel("Solar-X [arcsec]")
        ax.set_ylabel("Solar-Y [arcsec]")
        ax.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.3)

        save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"[繪圖完成] 向量疊加圖已輸出至: {os.path.abspath(save_path)}")

        if show:
            plt.show()
        else:
            plt.close()