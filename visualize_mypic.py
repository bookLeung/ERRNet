"""
Generate comparison visualizations for mypic self-collected photos.
For each image: Input | Baseline Output | Edge-Aware Output + Residual + Difference.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

RESULT_DIR = "results/mypic"
OUTPUT_DIR = "results/mypic/visualizations"
IMAGES = ["p1tree", "p2kobe", "p3air", "p4luxun", "p5badashanren"]

LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE = 24


def load_img(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def make_label(img_h, text, font_size=LABEL_FONT_SIZE):
    """Create a label strip to place above a sub-image."""
    from PIL import Image as PILImage
    label_h = font_size + 12
    label = PILImage.new("RGB", (1, label_h), color=(255, 255, 255))
    # We'll handle labels in the text rendering step
    return label_h


def np_to_pil(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def compute_residual(img, ref):
    """Amplified residual: |img - ref| * 5, clamped to [0,255]."""
    diff = np.abs(img - ref)
    diff_amp = np.clip(diff * 5, 0, 255)
    return diff_amp


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name in IMAGES:
        base_path = os.path.join(RESULT_DIR, "mypic_baseline", name)
        edge_path = os.path.join(RESULT_DIR, "mypic_edge", name)

        input_img = load_img(os.path.join(base_path, "m_input.png"))
        baseline_out = load_img(os.path.join(base_path, "baseline.png"))
        edge_out = load_img(os.path.join(edge_path, "edge_only.png"))

        # Ensure same size (crop to min dimensions)
        h = min(input_img.shape[0], baseline_out.shape[0], edge_out.shape[0])
        w = min(input_img.shape[1], baseline_out.shape[1], edge_out.shape[1])
        input_img = input_img[:h, :w, :]
        baseline_out = baseline_out[:h, :w, :]
        edge_out = edge_out[:h, :w, :]

        # Residual maps (amplified 5x for visibility)
        res_baseline = compute_residual(input_img, baseline_out)
        res_edge = compute_residual(input_img, edge_out)

        # Difference between baseline and edge (amplified 10x)
        diff_be = np.clip(np.abs(baseline_out - edge_out) * 10, 0, 255)

        # ---- Figure 1: Main comparison (3 columns) ----
        # Layout: 3 columns: Input | Baseline | Edge-Aware
        gap = 4
        label_h = LABEL_FONT_SIZE + 8
        title_h = TITLE_FONT_SIZE + 10
        total_h = title_h + label_h + h + gap * 2
        total_w = w * 3 + gap * 4

        main_fig = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255  # white bg

        # Place title
        # We'll use PIL for text rendering
        main_pil = Image.fromarray(main_fig)

        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", TITLE_FONT_SIZE)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", LABEL_FONT_SIZE)
        except Exception:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()

        draw = ImageDraw.Draw(main_pil)

        # Title
        draw.text((total_w // 2 - 150, 4), f"Self-Collected Photo: {name}", fill=(0, 0, 0), font=font_title)

        # Column headers
        col_labels = ["Input (Mixed)", "Baseline (ERRNet)", "Edge-Aware (Ours)"]
        for ci, label_text in enumerate(col_labels):
            x_start = gap + ci * (w + gap)
            text_w = draw.textlength(label_text, font=font_label) if hasattr(draw, 'textlength') else len(label_text) * 10
            draw.text(
                (x_start + (w - text_w) // 2, title_h),
                label_text,
                fill=(0, 0, 0),
                font=font_label,
            )

        # Place images
        images_list = [input_img, baseline_out, edge_out]
        for ci, img_arr in enumerate(images_list):
            x_start = gap + ci * (w + gap)
            y_start = title_h + label_h
            main_pil.paste(np_to_pil(img_arr), (x_start, y_start))

        main_pil.save(os.path.join(OUTPUT_DIR, f"{name}_comparison.png"))

        # ---- Figure 2: Residual + Difference (3 columns) ----
        res_h = h
        res_w = w * 3 + gap * 4
        res_total_h = title_h + label_h + h + gap * 2
        res_fig = Image.new("RGB", (res_w, res_total_h), (255, 255, 255))
        draw_res = ImageDraw.Draw(res_fig)

        draw_res.text(
            (res_w // 2 - 160, 4),
            f"Residual & Difference Analysis: {name}",
            fill=(0, 0, 0),
            font=font_title,
        )

        col_labels2 = [
            "|Input - Baseline| (x5)",
            "|Input - Edge| (x5)",
            "|Baseline - Edge| (x10)",
        ]
        for ci, label_text in enumerate(col_labels2):
            x_start = gap + ci * (w + gap)
            text_w = draw_res.textlength(label_text, font=font_label) if hasattr(draw_res, 'textlength') else len(label_text) * 10
            draw_res.text(
                (x_start + (w - text_w) // 2, title_h),
                label_text,
                fill=(0, 0, 0),
                font=font_label,
            )

        for ci, img_arr in enumerate([res_baseline, res_edge, diff_be]):
            x_start = gap + ci * (w + gap)
            y_start = title_h + label_h
            res_fig.paste(np_to_pil(img_arr), (x_start, y_start))

        res_fig.save(os.path.join(OUTPUT_DIR, f"{name}_residual.png"))

        print(f"✓ {name}: comparison + residual saved")

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
