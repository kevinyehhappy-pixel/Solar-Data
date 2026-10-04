from dataclasses import dataclass, field
import os
from typing import Dict, Optional, Tuple
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SolarDataROI:
    """模組二交付給下游分析的標準數據容器"""

    cube: np.ndarray  # 裁切後的 3D 數據立方體 [Frames, Y, X]
    wcs: WCS  # 裁切後對應的 2D 天文空間坐標 (WCS)
    header: fits.Header  # 關聯 Header
    scale_km_per_pix: float  # 本地像素物理尺度 (km/pixel)
    arcsec_per_pix: Tuple[float, float]  # (Y_scale, X_scale) 單位: arcsec/pixel
    center_hpc: Tuple[float, float]  # 裁切中心日心角秒 (Tx, Ty) 單位: arcsec
    radial_angle_deg: float  # 局部日面法線徑向向量夾角 (相對於水平 X 軸, 逆時針為正)
    location_type: str  # 區域屬性: 'disk' (盤面) 或 'limb' (邊緣)
    foreshortening_mu: float  # 投影補償因子 cos(theta)

    aux_layers: Dict[str, Optional[np.ndarray]] = field(
        default_factory=lambda: {"b_los": None, "doppler": None}
    )

    def attach_aux_layer(self, layer_name: str, data_2d: np.ndarray):
        if data_2d.shape != self.cube.shape[1:]:
            raise ValueError(
                f"附加圖層維度 {data_2d.shape} 與 SJI 空間維度 {self.cube.shape[1:]} 不一致！"
            )
        self.aux_layers[layer_name] = data_2d
        print(f"  - [擴充層] 已成功掛載 '{layer_name}' 至 ROI 容器。")


class AstrometryCalibrator:

    def __init__(self, data_cube: np.ndarray, header: fits.Header):
        self.data_cube = data_cube
        self.header = header

        full_wcs = WCS(self.header)
        if full_wcs.is_celestial:
            self.wcs_spatial = full_wcs
        else:
            self.wcs_spatial = full_wcs.celestial

    def calibrate_and_crop(
        self,
        pixel_bounds: Optional[Tuple[int, int, int, int]] = None,
        arcsec_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> SolarDataROI:
        print("\n[階段 1/3] 執行天文物理幾何定標...")
        n_frames, y_len, x_len = self.data_cube.shape

        # 1. 計算像素角尺度與公里尺度
        scales_deg = proj_plane_pixel_scales(self.wcs_spatial)
        cdelt_x_arcsec = float(scales_deg[0] * 3600.0)
        cdelt_y_arcsec = float(scales_deg[1] * 3600.0)

        dsun_meters = self.header.get("DSUN_OBS", 1.495978707e11)
        km_per_arcsec = (dsun_meters * (np.pi / (180.0 * 3600.0))) / 1000.0
        avg_arcsec_scale = (cdelt_x_arcsec + cdelt_y_arcsec) / 2.0
        scale_km_per_pix = avg_arcsec_scale * km_per_arcsec

        print(
            f"  - 像素角解析度 : X: {cdelt_x_arcsec:.3f}\", Y: {cdelt_y_arcsec:.3f}\" / pixel"
        )
        print(
            f"  - 物理板塊尺度 : {scale_km_per_pix:.2f} km/pixel (換算基準: {km_per_arcsec:.2f} km/arcsec)"
        )

        # 2. 確定裁切邊界
        if arcsec_bounds is not None:
            tx_min, tx_max, ty_min, ty_max = arcsec_bounds
            print(
                f"\n[階段 2/3] 執行角秒邊界裁切模式: [{tx_min}\" ~ {tx_max}\", {ty_min}\" ~ {ty_max}\"]"
            )
            # 角秒轉回度數傳入 WCS
            pix_bl = self.wcs_spatial.world_to_pixel_values(
                tx_min / 3600.0, ty_min / 3600.0
            )
            pix_tr = self.wcs_spatial.world_to_pixel_values(
                tx_max / 3600.0, ty_max / 3600.0
            )

            xmin = int(np.clip(np.floor(min(pix_bl[0], pix_tr[0])), 0, x_len))
            xmax = int(np.clip(np.ceil(max(pix_bl[0], pix_tr[0])), 0, x_len))
            ymin = int(np.clip(np.floor(min(pix_bl[1], pix_tr[1])), 0, y_len))
            ymax = int(np.clip(np.ceil(max(pix_bl[1], pix_tr[1])), 0, y_len))
        elif pixel_bounds is not None:
            ymin, ymax, xmin, xmax = pixel_bounds
            print(
                f"\n[階段 2/3] 執行像素邊界裁切模式: Y[{ymin}:{ymax}], X[{xmin}:{xmax}]"
            )
            ymin = max(0, ymin)
            ymax = min(y_len, ymax)
            xmin = max(0, xmin)
            xmax = min(x_len, xmax)
        else:
            print("\n[階段 2/3] 未指定邊界，預設保留全視場。")
            ymin, ymax, xmin, xmax = 0, y_len, 0, x_len

        cropped_cube = self.data_cube[:, ymin:ymax, xmin:xmax]
        cropped_wcs = self.wcs_spatial.slice(
            (slice(ymin, ymax), slice(xmin, xmax))
        )

        # 3. 幾何定位與日面法線計算 (修正度數轉角秒)
        print("[階段 3/3] 計算日面中心距離與法線方向基準...")
        crop_center_y = (ymax - ymin) / 2.0
        crop_center_x = (xmax - xmin) / 2.0

        # WCS 回傳單位為度 (Degrees)
        deg_x, deg_y = cropped_wcs.pixel_to_world_values(
            crop_center_x, crop_center_y
        )
        # 精確轉為日心角秒 (arcsec)
        center_tx = float(deg_x * 3600.0)
        center_ty = float(deg_y * 3600.0)

        r_arcsec = np.sqrt(center_tx**2 + center_ty**2)
        rsun_arcsec = self.header.get("RSUN_OBS", 959.63)

        if r_arcsec < 0.95 * rsun_arcsec:
            location_type = "disk"
            mu = np.sqrt(1.0 - (r_arcsec / rsun_arcsec) ** 2)
        else:
            location_type = "limb"
            mu = 0.0

        # 計算從太陽中心指向該區的法線向量夾角
        radial_angle_rad = np.arctan2(center_ty, center_tx)
        radial_angle_deg = float(np.degrees(radial_angle_rad))

        print(f"  - ROI 視場中心 : ({center_tx:.2f}\", {center_ty:.2f}\")")
        print(
            f"  - 日心距離     : {r_arcsec:.2f}\" (太陽視半徑: {rsun_arcsec:.2f}\")"
        )
        print(
            f"  - 區域分類     : {location_type.upper()} (投影係數 μ: {mu:.3f})"
        )
        print(
            f"  - 日面法線夾角 : {radial_angle_deg:.2f}° (作為針狀體風花圖 0° 參考軸)"
        )
        print(f"  - 裁切後維度   : {cropped_cube.shape} [Frames, Y, X]")

        return SolarDataROI(
            cube=cropped_cube,
            wcs=cropped_wcs,
            header=self.header,
            scale_km_per_pix=scale_km_per_pix,
            arcsec_per_pix=(cdelt_y_arcsec, cdelt_x_arcsec),
            center_hpc=(center_tx, center_ty),
            radial_angle_deg=radial_angle_deg,
            location_type=location_type,
            foreshortening_mu=float(mu),
        )

    @staticmethod
    def display_roi_preview(
        roi: SolarDataROI,
        frame_idx: int = 0,
        save_path: str = "./solar_data/roi_preview.png",
    ):
        """渲染科學影像並自動存檔，避免視窗阻塞問題"""
        print(f"\n[預覽輸出] 正在渲染並儲存 ROI 影像至 {save_path} ...")
        frame = roi.cube[frame_idx]
        vmin, vmax = np.percentile(frame, [1.0, 99.5])

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection=roi.wcs)

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

        # 繪製日面法線向量箭頭
        h, w = frame.shape
        cy, cx = h / 2.0, w / 2.0
        arrow_len = min(h, w) * 0.25
        rad = np.radians(roi.radial_angle_deg)
        dx = arrow_len * np.cos(rad)
        dy = arrow_len * np.sin(rad)

        ax.annotate(
            "",
            xy=(cx + dx, cy + dy),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle="->", color="yellow", lw=2, mutation_scale=20
            ),
        )
        ax.text(
            cx + dx,
            cy + dy,
            " Local Radial",
            color="yellow",
            fontsize=10,
            fontweight="bold",
        )

        ax.set_title(
            f"IRIS SJI 2796 Å [ROI Cropped]\n"
            f"Scale: {roi.scale_km_per_pix:.2f} km/pix | Type: {roi.location_type.upper()}",
            fontsize=12,
        )

        # 簡化軸標籤格式化，避免定位器計算過載
        try:
            lon = ax.coords[0]
            lat = ax.coords[1]
            lon.set_major_formatter("s.s")
            lat.set_major_formatter("s.s")
        except Exception:
            pass

        ax.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.4)
        plt.tight_layout()

        # 先儲存到硬碟，確保分析結果不丟失
        plt.savefig(save_path, dpi=200)
        print(f"  - [儲存成功] 預覽圖已寫入: {os.path.abspath(save_path)}")

        # 顯示視窗
        plt.show()