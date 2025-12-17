"""
PURPOSE: Generate flood maps and raster masks using Sentinel-1 SAR and Sentinel-2 optical imagery.

INPUTS: VV bands of Sentinel 1 image during peak and post flood dates. RGB (bands 2, 3, 4) of Monthly composite of Sentinel-2 for overlay.
Images were preprocessed (compositing, mosaicking, cloudmasking) on Google Earth Engine to lessen file size.
Bands were selected since streamlit crashes when too much data is loaded.

OUTPUTS: Flood mask geotiff. Visualization of flood masks during peak and post flooding.

FLOOD MASK PARAMETERS: Default values set in the GUI are the most optimal values.

ChatGPT 4 and 5 was used in creating a template for the Streamlit app and debugging errors regarding the download button, session_states, and matplotlib visualizations.

REFERENCES:
Lee filter for despeckling using scipy: Groff (2017). Despeckling Synthetic Aperture Radar (SAR) Images. https://www.kaggle.com/code/jgroff/despeckling-synthetic-aperture-radar-sar-images.
S1 Water thresholding: McVittie (2019). STEP ESA Flood mapping tutorial. https://step.esa.int/docs/tutorials/tutorial_s1floodmapping.pdf
Streamlit widgets documentation. https://docs.streamlit.io/develop.
"""

import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from streamlit_image_coordinates import streamlit_image_coordinates
from scipy.ndimage import uniform_filter
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Flood Mapping using Sentinel Imagery")
st.text("Generate flood maps and raster masks using Sentinel-1 SAR and Sentinel-2 optical imagery.")
st.caption("Created by Francine Soriano for IPA. Sample images show impacts of Typhoon Tino (4 Nov 2025) to Bago City, Philippines.")
st.caption("Images were clipped to not exceed streamlit's deployment processing limits for free tier! Bigger images can be loaded by using locally hosted version of GUI.")

# FUNCTIONS ---------------------------------------

def read_uploaded_raster(uploaded_file):
    with MemoryFile(uploaded_file.read()) as memfile:
        with memfile.open() as src:
            data = src.read()
            profile = src.profile
    # print("####### SENTINEL IMAGES AY", type(data), data.shape)
    return data, profile

def db_to_linear(db_array):
    """Convert dB to linear scale, skipping NaNs."""
    lin = np.full_like(db_array, np.nan, dtype=np.float32)
    mask = ~np.isnan(db_array)
    lin[mask] = 10 ** (db_array[mask] / 10)
    return lin

def linear_to_db(lin_array):
    """Convert linear to dB safely, avoiding log(0)."""
    db = np.full_like(lin_array, np.nan, dtype=np.float32)
    mask = ~np.isnan(lin_array)
    db[mask] = 10 * np.log10(np.maximum(lin_array[mask], 1e-8))
    return db

def lee_filter(img, size):
    """Lee speckle filter with NaN handling."""
    # Replace NaNs with 0 temporarily for convolution
    img_filled = np.nan_to_num(img, nan=0.0)
    
    # Local mean and variance
    mean = uniform_filter(img_filled, size)
    mean_sq = uniform_filter(img_filled**2, size)
    var = mean_sq - mean**2
    
    # Noise variance
    noise_var = np.nanmean(var)
    
    # Lee weighting
    w = var / (var + noise_var)
    result = mean + w * (img_filled - mean)
    
    # Restore NaNs for NoData
    result[np.isnan(img)] = np.nan
    return result

def mask_to_geotiff_bytes(mask, reference_profile):
    memfile = MemoryFile()
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="lzw"
    )

    with memfile.open(**profile) as dst:
        dst.write(mask.astype("uint8"), 1)

    return memfile.read()


# FILE UPLOADS -------------------------------------------------- 

st.header("Upload images")
st.subheader("Sentinel-1 Peak Flood")
s1_peak_file = st.file_uploader("The Sentinel-1 Image RIGHT AFTER the typhoon arrival date captures the areas that experienced peak flooding.", type=["tif", "tiff"], help="Sentinel-1 Image closest to typhoon arrival date")
s1_peak_date = st.date_input("Enter peak-flood image capture date", value="today")
st.subheader("Sentinel-1 Post Flood")
s1_post_file = st.file_uploader("The Sentinel-1 Image AFTER THE PEAK FLOOD captures the areas where flooding has not subsided yet.", type=["tif", "tiff"])
s1_post_date = st.date_input("Enter post-flood image capture date", value="today")
st.subheader("Sentinel 2 for Basemap")
s2_file   = st.file_uploader("Sentinel-2 RGB (Current version assumes that the raster only contains bands 2, 3, & 4 to minimize file size.)", type=["tif", "tiff"])

st.markdown("---")

st.header("Flood Mask Parameters")
vv_thresh = st.slider(
    "Select VV thresholds for Water (Default values have been set to the standard water threshold for Sentinel-1 VV images)",
    min_value=-50.0, max_value=1.0, value=(-20.0, -14.5),
    format="%0.3f"
)

mask_opacity = st.slider("Mask opacity", min_value=0.0, max_value=1.0, value=1.0)

gamma = st.slider("Gamma for Sentinel-2 (Lower gamma makes image brighter.)", min_value=0.0, max_value=1.0, value=0.35)

load_imgs = st.button("Load images", type="secondary")
despeckle = st.button("Despeckle images", type="secondary")
generate = st.button("Generate mask", type="primary")

st.markdown("---")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if not (s1_peak_file and s1_post_file and s2_file):
    st.info("Upload two Sentinel-1 VV images and one Sentinel-2 RGB image.")
    st.stop()

s1_1, s1_profile = read_uploaded_raster(s1_peak_file)
s1_2, _          = read_uploaded_raster(s1_post_file)
s2,   s2_profile = read_uploaded_raster(s2_file)

# assumes VV is first band
s1_peak = s1_1[0]
s1_post = s1_2[0]

# changes shape from (band,row,col) to (row,col,band); switches bands 1 and 3 to get RGB from BGR
s2_rgb = np.transpose(s2, (1, 2, 0))[:, :, ::-1] 
s2_rgb = s2_rgb/10000 # to scale values from 0 to 1

# TO MAINTAIN LOADED IMAGES ------------------------

if "show_loaded" not in st.session_state:
    st.session_state.show_loaded = False

if "show_despeckled" not in st.session_state:
    st.session_state.show_despeckled = False

# --------------------------------------------------
# TOP ROW: SENTINEL-1
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Sentinel-1 - {s1_peak_date} (VV)")
    fig1, ax1 = plt.subplots()
    ax1.axis("off")

with col2:
    st.subheader(f"Sentinel-1 - {s1_post_date} (VV)")
    fig2, ax2 = plt.subplots()
    ax2.axis("off")

st.text("Applying uniform filter to reduce speckle noise in S1 images.")
st.caption("Note that better despeckling can be achieved by using SNAP. The despeckling in this GUI using the uniform filter is for demonstration purposes only. Better despeckling will also result in better water masks.")
col3, col4 = st.columns(2)
with col3:
    # st.subheader(f"Sentinel-1 - {s1_peak_date} (VV)")
    fig3, ax3 = plt.subplots()
    ax3.axis("off")

with col4:
    # st.subheader(f"Sentinel-1 - {s1_peak_date} (VV)")
    fig4, ax4 = plt.subplots()
    ax4.axis("off")

s1_peak_despeckled = linear_to_db(lee_filter(db_to_linear(s1_peak), size=5))
s1_post_despeckled = linear_to_db(lee_filter(db_to_linear(s1_post), size=5))

# WHEN LOAD IMAGES BUTTON IS CLICKED --------------------------------------------------
if load_imgs:
    st.session_state.show_loaded = True

if st.session_state.show_loaded:
    with col1:
        ax1.imshow(s1_peak, cmap="Blues", vmin=-25, vmax=5)
        st.pyplot(fig1, clear_figure=True)
    
    with col2:
        ax2.imshow(s1_post, cmap="Blues", vmin=-25, vmax=5)
        st.pyplot(fig2, clear_figure=True)


# WHEN DESPECKLE BUTTON IS CLICKED --------------------------------------------------
if despeckle:
    st.session_state.show_despeckled = True

if st.session_state.show_despeckled:
    with col3:
        ax3.imshow(s1_peak_despeckled, cmap="Blues", vmin=-25, vmax=5)
        st.pyplot(fig3, clear_figure=False)
    
    with col4:
        ax4.imshow(s1_post_despeckled, cmap="Blues", vmin=-25, vmax=5)
        st.pyplot(fig4, clear_figure=False)

# --------------------------------------------------
# BOTTOM PANEL: INTERACTIVE DISPLAY
# --------------------------------------------------
st.subheader("Sentinel-2 with Sentinel-1 Flood Mask")

fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.axis("off")

# WHEN GENERATE BUTTON IS CLICKED --------------------------------------------------

mask_peak, mask_post, mask_rgb = None, None, None
mask_rgb_bytes = mask_to_geotiff_bytes(np.zeros(s1_peak.shape), s1_profile) # empty array para di magerror

if "show_mask" not in st.session_state:
    st.session_state.show_mask = False

if generate:
    st.session_state.show_mask = True

if st.session_state.show_mask:
    mask_peak = ( (s1_peak > vv_thresh[0]) & (s1_peak < vv_thresh[1]) ).astype(np.uint8) # VV VV > 0 AND VV < 0.025 
    mask_post = ( (s1_post > vv_thresh[0]) & (s1_post < vv_thresh[1]) ).astype(np.uint8)
    mask_rgb = mask_peak + mask_post*10
    
    # 1 = water in peak flood only
    # 10 = water in post flood only
    # 11 = water in both
    # 0 = no water
    
    mask_rgb = mask_peak + mask_post*10
    mask_rgb_bytes = mask_to_geotiff_bytes(mask_rgb, s1_profile)
    
    col3, col4 = st.columns(2)
    legend_elements = [Patch(facecolor="red", edgecolor="black", label="Water")]
    with col3:
        ax3.imshow(s1_peak_despeckled, cmap="Blues", vmin=-25, vmax=5)
        ax3.imshow(np.where(mask_peak == 1, 1, np.nan), cmap="Reds", vmin=0, vmax=1, alpha=mask_opacity)
        ax3.legend(handles=legend_elements, loc="lower right", frameon=True)
        st.pyplot(fig3, clear_figure=False) # needs to be false so that it shows up, since it uses same fig as despeckled
    
    with col4:
        ax4.imshow(s1_post_despeckled, cmap="Blues", vmin=-25, vmax=5)
        ax4.imshow(np.where(mask_post == 1, 1, np.nan), cmap="Reds", vmin=0, vmax=1, alpha=mask_opacity)
        ax4.legend(handles=legend_elements, loc="lower right", frameon=True)
        st.pyplot(fig4, clear_figure=False) # needs to be false so that it shows up, since it uses same fig as despeckled
    
    
    colors = ["red", "fuchsia", "yellow"]
    bounds = [0.5, 1.5, 10.5, 11.5]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    ax5.imshow(np.clip(np.power(s2_rgb, gamma), a_min=0.0, a_max=1.0)) # clip to avoid user warning from imshow clipping float to (0,1)
    ax5.imshow(np.where(mask_rgb > 0, mask_rgb, np.nan), cmap=cmap, norm=norm, alpha=mask_opacity)
    ax5.set_title(f"Flood Dynamics from {s1_peak_date} to {s1_post_date}")
    legend_elements = [
        Patch(facecolor="red", edgecolor="black", label="1 = Peak flood only"),
        Patch(facecolor="fuchsia", edgecolor="black", label="10 = Post-flood only"),
        Patch(facecolor="yellow", edgecolor="black", label="11 = Persistent water"),
    ]
    ax5.legend(handles=legend_elements, loc="lower right", frameon=True, title="Flood Class")
    st.pyplot(fig5, clear_figure=True)

    fig6, ax6 = plt.subplots(figsize=(4,3))
    ax6.hist(mask_rgb.flatten(), bins=[i for i in range(1,13)], edgecolor='black')
    ax6.set_title("Histogram of Flood Pixel Values")
    legend_elements_hist = [
        Patch(label="1 = Peak flood only"),
        Patch(label="10 = Post-flood only"),
        Patch(label="11 = Persistent water"),
    ]
    ax6.legend(handles=legend_elements_hist, loc="upper right", frameon=True, title="Flood Class")
    st.pyplot(fig6, width="content")


# DOWNLOAD FLOOD MASK BUTTON -----------------------------------

if mask_rgb is not None: # doublecheck
    mask_rgb_bytes = mask_to_geotiff_bytes(mask_rgb, s1_profile) 

st.download_button(
    label="Download Flood Mask Raster",
    data=mask_rgb_bytes,
    file_name="floodmask.tif",
    mime="image/tiff", # mime/mediatype standard
    type="primary")