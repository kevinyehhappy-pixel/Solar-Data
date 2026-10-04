from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib.pyplot as plt
import numpy as np
import sunpy.coordinates.sun as sun


@dataclass
class SolarDataROI:
    """模組二交付給下游分析的標準數據容器 (保留多通道擴充介面)"""

    cube: np.ndarray  # 裁切後的 3D 數據立方體 [Frames, Y, X]
    wcs: WCS  # 裁切後對應的 2D 天文空間坐標 (WCS)
    header: fits.Header  # 關聯 Header
    scale_km_per_pix: float  # 本地像素物理尺度 (km/pixel)
    arcsec_per_pix: Tuple[float, float]  # (Y_scale, X_scale) 單位: arcsec/pixel
    center_hpc: Tuple[float, float]  # 裁切中心日心角秒 (Tx, Ty)
    radial_angle_deg: float  # 局部日面法線徑向向量夾角 (相對於水平 X 軸, 逆時針為正)
    location_type: str  # 區域屬性: 'disk' (盤面) 或 'limb' (邊緣)
    foreshortening_mu: float  # 投影補償因子 cos(theta)

    # 預留擴充接口 (模組三、HMI 配準時直接掛載)
    aux_layers: Dict[str, Optional[np.ndarray]] = field(
        default_factory=lambda: {"b_los": None, "doppler": None}
    )

    def attach_aux_layer(self, layer_name: str, data_2d: np.ndarray):
        """掛載日後配準完成的 HMI 磁場或都卜勒速度場數據"""
        if data_2d.shape != self.cube.shape[1:]:
            raise ValueError(
                f"附加圖層維度 {data_2d.shape} 與 SJI 空間維度 {self.cube.shape[1:]} 不一致！"
            )
        self.aux_layers[layer_name] = data_2d
        print(f"  - [擴充層] 已成功掛載 '{layer_name}' 至 ROI 容器。")


class AstrometryCalibrator:
    """天文坐標解析、幾何物理定標與感興趣區域 (ROI) 裁切器"""

    def __init__(self, data_cube: np.ndarray, header: fits.Header):
        self.data_cube = data_cube
        self.header = header

        # 1. 建立 WCS 物件並精確提取二維天體空間坐標系 (Celestial WCS)
        full_wcs = WCS(self.header)
        if full_wcs.is_celestial:
            self.wcs_spatial = full_wcs
        else:
            # IRIS SJI 包含時間軸 (Axis 3)，呼叫 .celestial 自動抽離出純 (HPLN, HPLT) 2D 空間 WCS
            self.wcs_spatial = full_wcs.celestial

    def calibrate_and_crop(
        self,
        pixel_bounds: Optional[Tuple[int, int, int, int]] = None,
        arcsec_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> SolarDataROI:
        """執行幾何定標並進行雙模式 ROI 裁切

        :param pixel_bounds: (ymin, ymax, xmin, xmax) 像素坐標區間
        :param arcsec_bounds: (Tx_min, Tx_max, Ty_min, Ty_max) 日心角秒坐標區間
        """
        print("\n[階段 1/3] 執行天文物理幾何定標...")
        n_frames, y_len, x_len = self.data_cube.shape

        # ----------------------------------------------------
        # 1. 計算角秒尺度與日面物理公里尺度 (km/pixel)
        # ----------------------------------------------------
        # proj_plane_pixel_scales 取得單位為 degree/pixel
        scales_deg = proj_plane_pixel_scales(self.wcs_spatial)
        cdelt_x_arcsec = float(scales_deg[0] * 3600.0)
        cdelt_y_arcsec = float(scales_deg[1] * 3600.0)

        # 取得日地距離 (DSUN_OBS，公尺)；若無則調用天文常數 1 AU
        dsun_meters = self.header.get("DSUN_OBS", 1.495978707e11)
        # 1 弧度 = dsun_meters；1 角秒 = dsun_meters * (pi / (180 * 3600))
        km_per_arcsec = (dsun_meters * (np.pi / (180.0 * 3600.0))) / 1000.0
        avg_arcsec_scale = (cdelt_x_arcsec + cdelt_y_arcsec) / 2.0
        scale_km_per_pix = avg_arcsec_scale * km_per_arcsec

        print(
            f"  - 像素角解析度 : X: {cdelt_x_arcsec:.3f}\", Y: {cdelt_y_arcsec:.3f}\" / pixel"
        )
        print(
            f"  - 物理板塊尺度 : {scale_km_per_pix:.2f} km/pixel (換算基準: {km_per_arcsec:.2f} km/arcsec)"
        )

        # ----------------------------------------------------
        # 2. 確定裁切邊界 (Pixel Bounds)
        # ----------------------------------------------------
        if arcsec_bounds is not None:
            tx_min, tx_max, ty_min, ty_max = arcsec_bounds
            print(
                f"\n[階段 2/3] 執行角秒邊界裁切模式: [{tx_min}\" ~ {tx_max}\", {ty_min}\" ~ {ty_max}\"]"
            )
            # 將角秒反向投影回像素坐標
            pix_bl = self.wcs_spatial.world_to_pixel_values(tx_min, ty_min)
            pix_tr = self.wcs_spatial.world_to_pixel_values(tx_max, ty_max)

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

        # 執行數據陣列空間裁切
        cropped_cube = self.data_cube[:, ymin:ymax, xmin:xmax]

        # ----------------------------------------------------
        # 3. WCS 零點重構 (利用 Astropy slice 同步修正 CRPIX)
        # ----------------------------------------------------
        cropped_wcs = self.wcs_spatial.slice(
            (slice(ymin, ymax), slice(xmin, xmax))
        )

        # ----------------------------------------------------
        # 4. 幾何定位與日面法線夾角計算 (Local Radial Vector)
        # ----------------------------------------------------
        print("[階段 3/3] 計算日面中心距離與法線方向基準...")
        crop_center_y = (ymax - ymin) / 2.0
        crop_center_x = (xmax - xmin) / 2.0
        center_tx, center_ty = cropped_wcs.pixel_to_world_values(
            crop_center_x, crop_center_y
        )

        # 距離太陽球心 (0, 0) 的日心距離 r
        r_arcsec = np.sqrt(center_tx**2 + center_ty**2)
        rsun_arcsec = self.header.get("RSUN_OBS", 959.63)

        # 判定 Disk 或 Limb
        if r_arcsec < 0.95 * rsun_arcsec:
            location_type = "disk"
            # 投影補償因子 mu = cos(theta)
            mu = np.sqrt(1.0 - (r_arcsec / rsun_arcsec) ** 2)
        else:
            location_type = "limb"
            mu = 0.0

        # 計算日面法線向量角度 (從日心指向 ROI 中心的向量與 X 軸的正向夾角)
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
            center_hpc=(float(center_tx), float(center_ty)),
            radial_angle_deg=radial_angle_deg,
            location_type=location_type,
            foreshortening_mu=float(mu),
        )

    @staticmethod
    def display_roi_preview(roi: SolarDataROI, frame_idx: int = 0):
        """使用天文 WCSAxes 渲染校正與裁切後的科學影像"""
        frame = roi.cube[frame_idx]
        vmin, vmax = np.percentile(frame, [1.0, 99.5])

        fig = plt.figure(figsize=(9, 8))
        # 綁定天文坐標系投影
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

        # 繪製局部日面法線箭頭 (Local Radial Vector)
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
            " Local Radial (Normal)",
            color="yellow",
            fontsize=10,
            fontweight="bold",
        )

        ax.set_title(
            f"IRIS SJI 2796 Å [ROI Cropped]\n"
            f"Scale: {roi.scale_km_per_pix:.2f} km/pix | Type: {roi.location_type.upper()}",
            fontsize=12,
        )
        ax.set_xlabel("Solar-X (Helioprojective-Cartesian) [arcsec]")
        ax.set_ylabel("Solar-Y (Helioprojective-Cartesian) [arcsec]")
        ax.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        plt.show()