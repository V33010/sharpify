import math
import os
import sys

import numpy as np
from PIL import Image

# ----------------- USER-CONFIGURABLE CONSTANTS -----------------
# Physical screen (14" diagonal, 16:10 aspect ratio)
DIAG_IN = 14.0
INCH_TO_M = 0.0254
DIAG_M = DIAG_IN * INCH_TO_M
ASPECT_W = 16.0
ASPECT_H = 10.0

# Viewer geometry and eye model
S0 = 0.50  # nominal distance from eye to screen center (m)
A = 0.017  # reduced-eye image distance (retina) in meters (~17 mm)
PUPIL = 0.004  # pupil diameter (m). Adjust for light level (3-6 mm typical)

# Patch processing parameters
PATCH = 128  # patch size in pixels (square patches)
OVERLAP = 0.5  # fraction overlap (0.5 -> 50% overlap)
REGULARIZATION = 1e-3  # Wiener regularization factor (relative)
MIN_KERNEL_PIX = 1e-3  # below this kernel size (px) we treat PSF as delta

OUTPUT_DIR = "outputs"
# ---------------------------------------------------------------


def phys_screen_size_from_diag(diag_m, w_ratio=ASPECT_W, h_ratio=ASPECT_H):
    # returns (width_m, height_m)
    denom = math.sqrt(w_ratio**2 + h_ratio**2)
    width_m = diag_m * (w_ratio / denom)
    height_m = diag_m * (h_ratio / denom)
    return width_m, height_m


def make_disk_psf(h, w, radius_px):
    # Create a disk PSF of given radius (in pixels) centered in an array of size (h,w)
    # radius_px is radius in pixels (float). If radius_px < 0.5 returns delta.
    if radius_px <= 0.5:
        psf = np.zeros((h, w), dtype=np.float32)
        psf[h // 2, w // 2] = 1.0
        return psf

    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (r <= radius_px).astype(np.float32)
    s = mask.sum()
    if s <= 0:
        psf = np.zeros((h, w), dtype=np.float32)
        psf[h // 2, w // 2] = 1.0
        return psf
    return mask / s


def fft2(a):
    return np.fft.fft2(a)


def ifft2(a):
    return np.fft.ifft2(a)


def build_window(h, w):
    # Smooth 2D window to blend overlapping patches (outer product of 1D Hann)
    wy = np.hanning(h) if h > 1 else np.array([1.0])
    wx = np.hanning(w) if w > 1 else np.array([1.0])
    window = np.outer(wy, wx).astype(np.float32)
    # Avoid zero-sum windows
    window_sum = window.sum()
    if window_sum <= 0:
        return np.ones((h, w), dtype=np.float32)
    return window


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def process_channel(channel, width_m, height_m, s0, diopter_S):
    """
    channel: 2D float image (0..1)
    diopter_S: user-supplied diopter value (spectacle prescription convention)
    returns: processed channel (0..1)
    """
    h_img, w_img = channel.shape
    pixel_pitch_x = width_m / w_img
    pixel_pitch_y = height_m / h_img

    # patch stride
    step = int(PATCH * (1.0 - OVERLAP))
    if step < 1:
        step = 1

    out_acc = np.zeros_like(channel, dtype=np.float32)
    weight_acc = np.zeros_like(channel, dtype=np.float32)

    for y0 in range(0, h_img, step):
        for x0 in range(0, w_img, step):
            # compute patch extents and extract
            y1 = min(h_img, y0 + PATCH)
            x1 = min(w_img, x0 + PATCH)
            patch = channel[y0:y1, x0:x1].astype(np.float32)
            ph, pw = patch.shape

            # pad to PATCH if needed (we'll compute PSF in the padded domain)
            pad_h = PATCH - ph
            pad_w = PATCH - pw
            pad_top = 0
            pad_left = 0
            pad_bottom = pad_h
            pad_right = pad_w
            patch_padded = np.pad(
                patch, ((0, pad_bottom), (0, pad_right)), mode="reflect"
            )

            # compute physical coordinates (center of the patch) relative to screen center
            center_px_x = x0 + pw / 2.0
            center_px_y = y0 + ph / 2.0
            x_phys = (center_px_x - (w_img / 2.0)) * pixel_pitch_x
            y_phys = (center_px_y - (h_img / 2.0)) * pixel_pitch_y
            s_xy = math.sqrt(
                s0**2 + x_phys**2 + y_phys**2
            )  # distance from eye to patch center

            # compute local defocus in diopters (see derivation in comments)
            # Using the convention: diopter_S is the corrective lens power that would
            # correct the eye (i.e., standard spectacle prescription). Then the
            # local defocus (error) is:
            #    DeltaP(x,y) = 1/s(x,y) + diopter_S
            # (This follows from thin-lens relations and the algebra in the notes.)
            DeltaP = (1.0 / s_xy) + diopter_S

            # compute screen-domain PSF diameter: c_screen = p * s * |DeltaP|
            c_screen_m = PUPIL * s_xy * abs(DeltaP)
            # diameter in pixels (use x pitch for conversion; approximate square pixels)
            d_px = c_screen_m / pixel_pitch_x

            # bound and create PSF in padded domain
            # use radius = d_px/2
            radius_px = d_px / 2.0
            if radius_px < MIN_KERNEL_PIX:
                # effectively no blur; treat PSF as delta -> inverse is identity
                pre_patch = patch_padded
            else:
                psf = make_disk_psf(PATCH, PATCH, radius_px)
                # convert PSF to OTF via FFT; must shift PSF so its centre is at origin
                otf = fft2(np.fft.ifftshift(psf))

                # avoid tiny denominators: compute regularization K proportional to max(|otf|^2)
                power = np.abs(otf) ** 2
                K = REGULARIZATION * power.max()

                # FFT of the (padded) patch
                U = fft2(patch_padded)

                # Wiener-style approximate inverse operator: conj(otf)/( |otf|^2 + K )
                inv_op = np.conj(otf) / (power + K)

                # Apply inverse operator to the patch spectrum
                F_disp = U * inv_op
                pre_patch_full = np.real(ifft2(F_disp))

                pre_patch = pre_patch_full

            # unpad to original patch size
            pre_patch = pre_patch[:ph, :pw]

            # blend into accumulators using a smooth window
            win = build_window(ph, pw)
            out_acc[y0:y1, x0:x1] += pre_patch * win
            weight_acc[y0:y1, x0:x1] += win

    # normalize
    eps = 1e-12
    result = out_acc / (weight_acc + eps)
    # clip
    result = np.clip(result, 0.0, 1.0)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_path = input("Enter path to image file: ").strip('"')
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")
        return

    try:
        diopter = float(
            input(
                "Enter diopter (float, range -2.0 .. +2.0), negative=myopia, positive=hyperopia: "
            )
        )
    except Exception:
        print("Invalid diopter input. Must be a number.")
        return

    if diopter < -2.0 or diopter > 2.0:
        print("Diopter out of allowed range (-2 .. +2). Exiting.")
        return

    # load image
    im = Image.open(img_path).convert("RGB")
    w_px, h_px = im.size
    print(f"Loaded image {img_path} size {w_px}x{h_px}")

    width_m, height_m = phys_screen_size_from_diag(DIAG_M, ASPECT_W, ASPECT_H)
    print(f"Using screen physical size: {width_m*1000:.1f} mm x {height_m*1000:.1f} mm")
    print(f"Viewer distance (center): {S0*1000:.0f} mm ; pupil: {PUPIL*1000:.1f} mm")

    arr = np.asarray(im).astype(np.float32) / 255.0
    channels = [arr[..., i] for i in range(3)]

    print("Processing channels (this may take a while)...")
    out_channels = []
    for idx, ch in enumerate(channels):
        print(f" - channel {idx+1}/3")
        processed = process_channel(ch, width_m, height_m, S0, diopter)
        out_channels.append(processed)

    out_arr = np.stack(out_channels, axis=2)
    out_img = Image.fromarray((np.clip(out_arr, 0.0, 1.0) * 255.0).astype(np.uint8))

    base = os.path.basename(img_path)
    out_name = os.path.join(OUTPUT_DIR, "output-" + base)
    out_img.save(out_name)

    print(f"Saved pre-distorted image to: {out_name}")

    if diopter > 0:
        print(
            "NOTE: Input diopter is positive (hyperopia). Software-only correction for"
        )
        print(
            "true optical convergence is not possible on a conventional display; this"
        )
        print("pre-distortion attempts to improve perceived contrast but cannot create")
        print(
            "the extra physical convergence a weak eye requires. For hyperopia, consider"
        )
        print(
            "optical methods (glasses, contact lenses, or wavefront-capable displays)."
        )


if __name__ == "__main__":
    main()
