import io
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
import segmentation_models_pytorch as smp

# ---------------------------
# Configuration
# ---------------------------
IMG_SIZE = 256
PRED_THRESH = 0.5           # probability threshold for binary mask
PIXEL_TO_MM = 0.13          # circumference scaling factor (edit for your setup)
DEFAULT_MODEL_PATH = "best_model_convnext.pth"   # <-- updated checkpoint name

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Utility functions
# ---------------------------
def load_image_to_cv2(file) -> np.ndarray:
    """Load uploaded file into grayscale OpenCV image."""
    bytes_data = file.read()
    img_pil = Image.open(io.BytesIO(bytes_data)).convert("L")  # force grayscale
    img_np = np.array(img_pil)
    return img_np


def preprocess_image(img_np: np.ndarray):
    """
    Resize to 256x256, normalize to [0, 1],
    and convert to tensor shape (1, 1, H, W).
    Returns tensor and resized image for visualization.
    """
    img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))

    img_norm = img_resized.astype("float32") / 255.0
    img_norm = np.expand_dims(img_norm, axis=0)  # (1, H, W)
    img_norm = np.expand_dims(img_norm, axis=0)  # (1, 1, H, W)
    img_tensor = torch.from_numpy(img_norm).to(DEVICE)

    return img_tensor, img_resized


def postprocess_mask(prob_map: torch.Tensor, threshold: float = PRED_THRESH) -> np.ndarray:
    """
    prob_map: (1, 1, H, W) tensor after sigmoid
    Returns binary mask (H, W) uint8 with values {0, 1}
    """
    probs = prob_map.detach().cpu().numpy()
    mask = (probs > threshold).astype(np.uint8)[0, 0]  # (H, W)
    return mask


def calculate_head_circumference(mask_np: np.ndarray, pixel_to_mm: float = PIXEL_TO_MM) -> float:
    """
    Approximate head circumference as contour perimeter (in pixels)
    multiplied by pixel_to_mm scaling factor.
    mask_np: (H, W) binary mask {0,1}
    """
    mask_u8 = (mask_np * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    perimeter_px = cv2.arcLength(largest, True)
    circumference_mm = float(perimeter_px * pixel_to_mm)
    return circumference_mm


def overlay_mask_on_image(image_gray: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """
    Create a color overlay: original grayscale in background,
    mask in red on top.
    """
    img_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    mask_u8 = (mask_np * 255).astype(np.uint8)
    mask_color = np.zeros_like(img_color)
    mask_color[:, :, 2] = mask_u8  # red channel

    alpha = 0.6
    beta = 0.4
    overlay = cv2.addWeighted(img_color, alpha, mask_color, beta, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def overlay_cam_on_image(image_gray: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on grayscale image.
    cam: (H, W) float32 in [0,1]
    """
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    img_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    
    heatmap_resized = cv2.resize(heatmap, (img_color.shape[1], img_color.shape[0]))

    alpha = 0.5
    beta = 0.5
    overlay = cv2.addWeighted(img_color, alpha, heatmap_resized, beta, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


# ---------------------------
# Grad-CAM for segmentation
# ---------------------------
class SegmentationGradCAM:
    """
    Simple Grad-CAM for SMP U-Net segmentation model.
    We hook into the last decoder block.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # out: (N, C, H, W)
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0]: (N, C, H, W)
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate_cam(self, x: torch.Tensor) -> np.ndarray:
        """
        x: input tensor (1, 1, H, W)
        Returns Grad-CAM heatmap as (H, W) float32 in [0, 1]
        """
        self.model.zero_grad()

        # Forward
        logits = self.model(x)  # (1, 1, H, W)
        probs = torch.sigmoid(logits)

        # Use mean foreground probability as target
        target = probs.mean()
        target.backward()

        acts = self.activations       # (1, C, H', W')
        grads = self.gradients        # (1, C, H', W')

        if acts is None or grads is None:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # Global average pooling over spatial dimensions for weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = torch.relu(cam)

        # Normalize
        cam = cam.squeeze(0).squeeze(0)  # (H', W')
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # Resize to model output size
        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        cam_resized = cv2.resize(cam_np, (IMG_SIZE, IMG_SIZE))

        return cam_resized


# ---------------------------
# Model loading (cached)
# ---------------------------
@st.cache_resource
def load_model(model_path: str):
    model = smp.Unet(
        encoder_name="mit_b2",        # keep same encoder as training
        encoder_weights="imagenet",   # as requested
        in_channels=1,
        classes=1,
        activation=None
    ).to(DEVICE)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        st.sidebar.success(f"Loaded model weights from: {model_path}")
    else:
        st.sidebar.error(f"Model checkpoint not found at: {model_path}")
        st.sidebar.info(
            "Please export your trained weights to this path or update the checkpoint path "
            "in the sidebar. Without weights, predictions will not be meaningful."
        )

    model.eval()
    return model


# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(
        page_title="Fetal Head Segmentation & HC Estimation",
        page_icon="🩺",
        layout="wide"
    )

    # --- Custom CSS for healthcare-themed look ---
    st.markdown(
    """
    <style>
    .main {
        background: #f2f7fc;
    }
    .title-text {
        font-size: 32px;
        font-weight: 700;
        color: #1f3b57;
    }
    .subtitle-text {
        font-size: 16px;
        color: #4f6b88;
    }

    /* FIXED: readable text inside cards */
    .metric-card {
        padding: 1rem;
        border-radius: 0.75rem;
        background: #e9f2fc;  
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        border: 1px solid #c7d7eb;

        color: #0f1c2e;             /* NEW: dark navy text */
        font-size: 15px;            /* slightly larger for readability */
        font-weight: 500;           /* medium weight */
    }

    .metric-card b {
        color: #0a2540 !important;  /* bold headers inside cards now visible */
    }

    .footer-text {
        font-size: 12px;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True
)


    # --- Header Section ---
    col_header_left, col_header_right = st.columns([3, 2])

    with col_header_left:
        st.markdown('<div class="title-text">Fetal Head Segmentation & Biometry</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="subtitle-text">
            Real-time <b>MiT-B2 U-Net</b>–based fetal head segmentation with automatic  
            <b>Head Circumference (HC)</b> estimation and <b>Grad-CAM explainability</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_header_right:
        st.markdown(
            """
            <div class="metric-card">
            <b>Clinical Use Disclaimer</b><br/>
            This demo is for research and educational purposes only and  
            is <b>not validated</b> for direct clinical decision making.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")

    global PIXEL_TO_MM, PRED_THRESH
    PRED_THRESH = st.sidebar.slider("Mask Threshold", 0.1, 0.9, PRED_THRESH, 0.05)
    PIXEL_TO_MM = st.sidebar.number_input(
        "Pixel → mm (scale)",
        value=float(PIXEL_TO_MM),
        step=0.01,
        format="%.3f"
    )

    show_gradcam = st.sidebar.checkbox("Enable Grad-CAM visualization", value=True)

    st.sidebar.subheader("Model")
    model_path = st.sidebar.text_input(
        "Model checkpoint path (.pth)",
        value=DEFAULT_MODEL_PATH,
        help="Path to your trained MiT-B2 U-Net weights."
    )
    st.sidebar.caption("Backbone: MiT-B2 (SegFormer encoder) with ImageNet pre-training")

    model = load_model(model_path)

    # --- Upload Section ---
    st.subheader("1️⃣ Upload Ultrasound Image")

    uploaded_file = st.file_uploader(
        "Upload a fetal head ultrasound image (PNG/JPG/BMP/TIFF)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="Preferably a standard HC plane image."
    )

    if uploaded_file is None:
        st.info("Please upload a grayscale fetal ultrasound image to begin.")
        return

    # Read original image
    img_np = load_image_to_cv2(uploaded_file)

    col_orig, col_info = st.columns([3, 2])
    with col_orig:
        st.image(img_np, caption="Uploaded Ultrasound (grayscale)")
    with col_info:
        st.markdown(
            """
            <div class="metric-card">
            <b>Tips for best results</b>
            <ul>
              <li>Use a standard HC plane with clear skull boundary.</li>
              <li>Avoid heavy noise or motion artifacts.</li>
              <li>Ensure head occupies a significant region of the frame.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("2️⃣ Segmentation & Head Circumference Estimation")

    # Preprocess
    img_tensor, img_resized = preprocess_image(img_np)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)

    mask_np = postprocess_mask(probs, threshold=PRED_THRESH)
    hc_mm = calculate_head_circumference(mask_np, pixel_to_mm=PIXEL_TO_MM)
    overlay_rgb = overlay_mask_on_image(img_resized, mask_np)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Predicted Segmentation Mask (256×256)**")
        st.image(mask_np * 255)

    with col2:
        st.markdown("**Overlay: Mask on Ultrasound Image**")
        st.image(overlay_rgb)

    # Metric card
    st.markdown("")
    col_metric, col_note = st.columns([1.2, 2])

    with col_metric:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Estimated Head Circumference (HC)**")
        st.metric(label="HC (approx.)", value=f"{hc_mm:.2f} mm")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_note:
        st.markdown(
            """
            <div class="metric-card">
            <b>Interpretation Note</b><br/>
            HC is approximated from the largest skull contour perimeter in the predicted mask,
            scaled by the pixel-to-mm factor you defined in the sidebar.  
            Make sure the scaling matches your ultrasound system's calibration.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Grad-CAM Section ---
    st.markdown("---")
    st.subheader("3️⃣ Explainability: Grad-CAM Heatmap")

    if show_gradcam:
        with st.expander("What does Grad-CAM show?", expanded=True):
            st.write(
                "Grad-CAM highlights spatial regions that most strongly influence the model's prediction. "
                "Warm colors correspond to regions where the model is focusing while segmenting the fetal head."
            )

        # Build Grad-CAM object on the last decoder block
        target_layer = model.decoder.blocks[-1]
        grad_cam = SegmentationGradCAM(model, target_layer)

        # Need gradient, so no torch.no_grad() here
        cam_map = grad_cam.generate_cam(img_tensor)
        grad_cam.remove_hooks()

        cam_overlay = overlay_cam_on_image(img_resized, cam_map)

        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            st.markdown("**Grad-CAM Heatmap (normalized)**")
            st.image(cam_map, clamp=True)

        with col_cam2:
            st.markdown("**Grad-CAM Overlay on Ultrasound**")
            st.image(cam_overlay)

    else:
        st.info("Enable Grad-CAM in the sidebar to visualize model attention.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer-text">
        © Fetal Head Segmentation Demo · Research prototype – not for clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
