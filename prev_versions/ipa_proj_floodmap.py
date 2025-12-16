import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from streamlit_image_coordinates import streamlit_image_coordinates

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Sentinel-1 Mask over Sentinel-2 (Local Streamlit App)")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def read_uploaded_raster(uploaded_file):
    with MemoryFile(uploaded_file.read()) as memfile:
        with memfile.open() as src:
            data = src.read()
            profile = src.profile
    return data, profile

def normalize(img, pmin=2, pmax=98):
    vmin = np.nanpercentile(img, pmin)
    vmax = np.nanpercentile(img, pmax)
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)

# --------------------------------------------------
# SIDEBAR: FILE UPLOADS
# --------------------------------------------------
st.sidebar.header("Upload images")

s1_file_1 = st.sidebar.file_uploader("Sentinel-1 VV - Date 1", type=["tif", "tiff"])
s1_file_2 = st.sidebar.file_uploader("Sentinel-1 VV - Date 2", type=["tif", "tiff"])
s2_file   = st.sidebar.file_uploader("Sentinel-2 RGB", type=["tif", "tiff"])

st.sidebar.markdown("---")

vv_thresh = st.sidebar.slider(
    "VV threshold (normalized)",
    0.0, 1.0, 0.2, 0.01
)

mask_opacity = st.sidebar.slider(
    "Mask opacity",
    0.0, 1.0, 0.5, 0.05
)

generate = st.sidebar.button("Generate mask")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if not (s1_file_1 and s1_file_2 and s2_file):
    st.info("Upload two Sentinel-1 VV images and one Sentinel-2 RGB image.")
    st.stop()

s1_1, s1_profile = read_uploaded_raster(s1_file_1)
s1_2, _          = read_uploaded_raster(s1_file_2)
s2,   s2_profile = read_uploaded_raster(s2_file)

s1_1 = s1_1[0]
s1_2 = s1_2[0]
s2_rgb = np.transpose(s2[:3], (1, 2, 0))

s1_1_n = normalize(s1_1)
s1_2_n = normalize(s1_2)
s2_n   = normalize(s2_rgb)

# --------------------------------------------------
# TOP ROW: SENTINEL-1
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentinel-1 - Date 1 (VV)")
    fig1, ax1 = plt.subplots()
    ax1.imshow(s1_1_n, cmap="gray")
    ax1.axis("off")
    st.pyplot(fig1)

with col2:
    st.subheader("Sentinel-1 - Date 2 (VV)")
    fig2, ax2 = plt.subplots()
    ax2.imshow(s1_2_n, cmap="gray")
    ax2.axis("off")
    st.pyplot(fig2)

# --------------------------------------------------
# BOTTOM PANEL: INTERACTIVE DISPLAY
# --------------------------------------------------
st.subheader("Sentinel-2 with Sentinel-1 Mask (Zoom & Pan Enabled)")

mask = None
mask_rgba = None

if generate:
    mask = (s1_1_n < vv_thresh) & (s1_2_n < vv_thresh)

    mask_rgba = np.zeros((*mask.shape, 4))
    mask_rgba[..., 0] = 1.0
    mask_rgba[..., 3] = mask * mask_opacity

# Create composite image
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.imshow(s2_n)
if mask_rgba is not None:
    ax3.imshow(mask_rgba)
ax3.axis("off")

# Interactive zoomable display
coords = streamlit_image_coordinates(fig3, key="zoom_view")

# --------------------------------------------------
# EXPORTS
# --------------------------------------------------
st.markdown("### Export")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    if generate:
        fig3.canvas.draw()
        png_path = "sentinel1_mask_overlay.png"
        fig3.savefig(png_path, dpi=300)

        with open(png_path, "rb") as f:
            st.download_button(
                "Export displayed image (PNG)",
                f,
                file_name="sentinel1_mask_on_sentinel2.png",
                mime="image/png"
            )

with col_exp2:
    if generate:
        mask_tif_path = "sentinel1_mask.tif"

        profile = s1_profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress="lzw"
        )

        with rasterio.open(mask_tif_path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)

        with open(mask_tif_path, "rb") as f:
            st.download_button(
                "Export mask as GeoTIFF",
                f,
                file_name="sentinel1_mask.tif",
                mime="image/tiff"
            )
