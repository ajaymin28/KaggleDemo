import gradio as gr
import numpy as np
import cv2
import time
from PIL import Image

STATE = {
    "corners": [],
    "subjects": [],
    "velocities": [],
    "grid_shape": (6, 12),
    "image": None,
    "cache": None,  # to store precomputed overlays/masks
    "animating": False,
}

def blank_canvas():
    img = np.ones((600, 1200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (1150, 550), (0, 180, 0), 10)
    return img

def precompute_masks_and_backgrounds(corners, grid_shape, img_shape, radar_shape=(1000,500)):
    n, m = grid_shape
    h_img, w_img = img_shape[:2]
    W, H = radar_shape

    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(src, dst)
    Hmat_inv = cv2.getPerspectiveTransform(dst, src)

    # Field grid for original image (with alpha)
    floor_grid_img = np.zeros((h_img, w_img, 4), dtype=np.uint8)
    radar_bg = np.zeros((H, W, 3), dtype=np.uint8) + 30
    # radar_bg = cv2.warpPerspective(STATE["image"], Hmat, (W, H))
    # corrected_bg = np.zeros((H, W, 3), dtype=np.uint8) + 255
    corrected_bg = cv2.warpPerspective(STATE["image"], Hmat, (W, H))

    radar_cell_mask = np.full((H, W), -1, dtype=np.int32)
    corrected_cell_mask = np.full((H, W), -1, dtype=np.int32)

    grid_cells_img = []
    radar_cells = []
    corrected_cells = []

    cell_w, cell_h = W // m, H // n
    idx = 0
    for i in range(n):
        for j in range(m):
            tl = (j*cell_w, i*cell_h)
            br = ((j+1)*cell_w, (i+1)*cell_h)
            cell_poly = np.array([
                tl, (br[0], tl[1]), br, (tl[0], br[1])
            ], dtype=np.int32)
            # Draw grid lines for floor image (perspective transform)
            cell_poly_img = cv2.perspectiveTransform(cell_poly[None].astype(np.float32), Hmat_inv)[0].astype(np.int32)
            cv2.polylines(floor_grid_img, [cell_poly_img], True, (0,0,0,160), 1)
            cv2.polylines(radar_bg, [cell_poly], True, (80,80,80), 1)
            cv2.polylines(corrected_bg, [cell_poly], True, (0,0,0), 1)
            # For radar/corrected cell mask
            cv2.fillPoly(radar_cell_mask, [cell_poly], idx)
            cv2.fillPoly(corrected_cell_mask, [cell_poly], idx)
            grid_cells_img.append(cell_poly_img)
            radar_cells.append(cell_poly)
            corrected_cells.append(cell_poly)
            idx += 1

    return {
        "floor_grid_img": floor_grid_img,
        "radar_bg": radar_bg,
        "radar_cell_mask": radar_cell_mask,
        "corrected_bg": corrected_bg,
        "corrected_cell_mask": corrected_cell_mask,
        "grid_cells_img": grid_cells_img,
        "radar_cells": radar_cells,
        "corrected_cells": corrected_cells,
        "Hmat": Hmat,
        "Hmat_inv": Hmat_inv,
        "radar_shape": (W,H),
    }

def refresh_cache():
    # Only if corners and image are valid
    if len(STATE["corners"]) == 4 and STATE["image"] is not None:
        STATE["cache"] = precompute_masks_and_backgrounds(
            STATE["corners"], STATE["grid_shape"], STATE["image"].shape)
    else:
        STATE["cache"] = None

def get_highlights(subjects, grid_cells_img):
    highlights = np.zeros(len(grid_cells_img), dtype=bool)
    for pt in subjects:
        for idx, cell in enumerate(grid_cells_img):
            if cv2.pointPolygonTest(cell, tuple(pt), False) > 0:
                highlights[idx] = True
    return highlights

def fast_overlay(img, cache, highlights, subjects):
    base = img.copy()
    floor_grid_img = cache["floor_grid_img"]
    grid_cells_img = cache["grid_cells_img"]
    radar_bg_cache = cache["radar_bg"]
    radar_cell_mask = cache["radar_cell_mask"]
    corrected_bg_cache = cache["corrected_bg"]
    # corrected_cell_mask = cache["corrected_cell_mask"]
    Hmat = cache["Hmat"]

    # --- Floor image ---
    if floor_grid_img.shape[2] == 4:
        a = floor_grid_img[...,3:] / 255.0
        base = (base*(1-a) + floor_grid_img[...,:3]*a).astype(np.uint8)
    highlight_mask = np.zeros_like(base)
    for idx, cell in enumerate(grid_cells_img):
        if highlights[idx]:
            cv2.fillPoly(highlight_mask, [cell], (0,0,255))
    base = cv2.addWeighted(base, 1.0, highlight_mask, 0.22, 0)
    for pt in subjects:
        cv2.circle(base, tuple(np.int32(pt)), 10, (0,0,255), -1)
    floor_pil = Image.fromarray(base)

    # --- Radar ---
    radar_bg = radar_bg_cache.copy()
    occupied_cells = np.flatnonzero(highlights)
    radar_mask = np.isin(radar_cell_mask, occupied_cells)
    radar_bg[radar_mask] = [0,0,255]
    if len(subjects):
        pts = np.array(subjects, dtype=np.float32)[None]
        pts_radar = cv2.perspectiveTransform(pts, Hmat)[0]
        for p in pts_radar:
            cv2.circle(radar_bg, tuple(np.clip(p.astype(int), [0,0], [radar_bg.shape[1]-1, radar_bg.shape[0]-1])), 10, (0,255,0), -1)
    radar_pil = Image.fromarray(radar_bg)

    # --- Perspective-corrected view (this is the fix) ---
    corrected_bg = corrected_bg_cache.copy()  # always fresh copy!
    # corrected_mask = np.isin(corrected_cell_mask, occupied_cells)
    # corrected_bg[corrected_mask] = [0,0,255]
    if len(subjects):
        pts = np.array(subjects, dtype=np.float32)[None]
        pts_corr = cv2.perspectiveTransform(pts, Hmat)[0]
        for p in pts_corr:
            cv2.circle(corrected_bg, tuple(np.clip(p.astype(int), [0,0], [corrected_bg.shape[1]-1, corrected_bg.shape[0]-1])), 10, (0,255,0), -1)
    corrected_pil = Image.fromarray(corrected_bg)

    return floor_pil, radar_pil, corrected_pil

def overlay_info(img, corners, subjects, grid_shape):
    if len(corners) == 4 and img is not None:
        if STATE["cache"] is None:
            refresh_cache()
        highlights = get_highlights(subjects, STATE["cache"]["grid_cells_img"])
        return fast_overlay(img, STATE["cache"], highlights, subjects)
    else:
        # No corners: just show basic images
        # print(len(corners), img)
        # print("using blank backgrounds")
        H, W = 500, 1000
        floor_pil = Image.fromarray(img if img is not None else blank_canvas())
        radar_pil = Image.fromarray(np.zeros((H, W, 3), np.uint8) + 30)
        corrected_pil = Image.fromarray(np.zeros((H, W, 3), np.uint8) + 255)
        return floor_pil, radar_pil, corrected_pil

def gradio_click_handler(evt: gr.SelectData):
    x, y = evt.index
    if len(STATE["corners"]) < 4:
        STATE["corners"].append((x, y))
        refresh_cache()
    else:
        STATE["subjects"].append([float(x), float(y)])
        # assign a random direction for new subject
        angle = np.random.uniform(0, 2*np.pi)
        v = [np.cos(angle), np.sin(angle)]
        STATE["velocities"].append(v)
    # print(f"Corner len {len(STATE['corners'])}")
    return overlay_info(
        STATE["image"], STATE["corners"], STATE["subjects"], STATE["grid_shape"]
    )

def reset_corners():
    STATE["corners"].clear()
    STATE["subjects"].clear()
    STATE["velocities"].clear()
    STATE["cache"] = None
    return update_canvas()

def clear_subjects():
    STATE["subjects"].clear()
    STATE["velocities"].clear()
    return overlay_info(
        STATE["image"], STATE["corners"], STATE["subjects"], STATE["grid_shape"]
    )

def set_grid(n, m):
    STATE["grid_shape"] = (n, m)
    refresh_cache()
    return overlay_info(
        STATE["image"], STATE["corners"], STATE["subjects"], STATE["grid_shape"]
    )

def update_canvas():
    if STATE["image"] is None:
        img = blank_canvas()
        STATE["image"] = img
    else:
        img = STATE["image"]
    return overlay_info(
        img, STATE["corners"], STATE["subjects"], STATE["grid_shape"]
    )

def on_image_upload(image):
    if image is not None:
        STATE["image"] = image.copy()
        STATE["corners"].clear()
        STATE["subjects"].clear()
        STATE["velocities"].clear()
        STATE["cache"] = None
        return overlay_info(
            STATE["image"], STATE["corners"], STATE["subjects"], STATE["grid_shape"]
        )
    else:
        img = blank_canvas()
        STATE["image"] = img
        STATE["corners"].clear()
        STATE["subjects"].clear()
        STATE["velocities"].clear()
        STATE["cache"] = None
        return overlay_info(
            img, STATE["corners"], STATE["subjects"], STATE["grid_shape"]
        )

def in_polygon(pt, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), tuple(pt), False) >= 0

def move_subjects():
    speed = 2.2
    direction_change_prob = 0.03
    max_steer_angle = np.pi / 16
    for i, pt in enumerate(STATE["subjects"]):
        vx, vy = STATE["velocities"][i]
        if np.random.rand() < direction_change_prob:
            angle = np.arctan2(vy, vx)
            angle += np.random.uniform(-max_steer_angle, max_steer_angle)
            vx, vy = np.cos(angle), np.sin(angle)
        norm = np.linalg.norm([vx, vy])
        if norm == 0:
            vx, vy = 1, 0
        vx, vy = vx / norm, vy / norm
        new_pt = [pt[0] + vx * speed, pt[1] + vy * speed]
        if len(STATE["corners"]) == 4 and not in_polygon(new_pt, STATE["corners"]):
            delta = 2
            for dtheta in np.linspace(0, 2 * np.pi, 8):
                probe = [pt[0] + delta * np.cos(dtheta), pt[1] + delta * np.sin(dtheta)]
                if in_polygon(probe, STATE["corners"]):
                    normal = np.array([probe[0] - pt[0], probe[1] - pt[1]])
                    normal = normal / (np.linalg.norm(normal) + 1e-6)
                    break
            else:
                normal = np.array([1.0, 0.0])
            v = np.array([vx, vy])
            v_reflect = v - 2 * (v @ normal) * normal
            v_reflect += np.random.uniform(-0.1, 0.1, size=2)
            v_reflect = v_reflect / (np.linalg.norm(v_reflect) + 1e-6)
            vx, vy = v_reflect[0], v_reflect[1]
            new_pt = [pt[0] + vx * speed, pt[1] + vy * speed]
            if not in_polygon(new_pt, STATE["corners"]):
                new_pt = pt
        STATE["subjects"][i] = new_pt
        STATE["velocities"][i] = [vx, vy]

def animate():
    STATE["animating"] = True
    while STATE["animating"]:
        if len(STATE["subjects"]) > 0 and len(STATE["corners"]) == 4:
            move_subjects()
        imgs = overlay_info(
            STATE["image"], STATE["corners"], STATE["subjects"], STATE["grid_shape"]
        )
        yield imgs
        time.sleep(0.015)  # ~60 FPS

def stop_animation():
    STATE["animating"] = False

with gr.Blocks() as demo:
    gr.Markdown("### ⚡ Super-fast Floor-to-Radar Grid Simulation")

    with gr.Row():
        with gr.Column():
            n = gr.Slider(2, 24, value=6, step=1, label="Grid Rows (N)")
            m = gr.Slider(2, 48, value=12, step=1, label="Grid Cols (M)")
            reset_btn = gr.Button("Reset Floor Corners")
            clear_btn = gr.Button("Clear Subjects")
            floor_img = gr.Image(type="numpy", interactive=True, label="Floor (click 4 corners, then players)")
            start_btn = gr.Button("Start Animation")
            stop_btn = gr.Button("Stop Animation")
        with gr.Column():
            radar_img = gr.Image(type="pil", interactive=False, label="Radar (topdown)")
            corrected_img = gr.Image(type="pil", interactive=False, label="Perspective-corrected Topdown")

    floor_img.select(gradio_click_handler, None, [floor_img, radar_img, corrected_img])
    floor_img.upload(on_image_upload, floor_img, [floor_img, radar_img, corrected_img])
    n.change(set_grid, [n, m], [floor_img, radar_img, corrected_img])
    m.change(set_grid, [n, m], [floor_img, radar_img, corrected_img])
    reset_btn.click(reset_corners, None, [floor_img, radar_img, corrected_img])
    clear_btn.click(clear_subjects, None, [floor_img, radar_img, corrected_img])
    start_btn.click(animate, None, [floor_img, radar_img, corrected_img], show_progress=False)
    stop_btn.click(stop_animation, None, None)
    demo.load(update_canvas, None, [floor_img, radar_img, corrected_img])

demo.queue().launch()
