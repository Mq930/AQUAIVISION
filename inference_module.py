"""
inference_module.py
Core processing module for AQUAVISION.
Lazy-loads TRT models on first use to avoid blocking server startup.
"""

import cv2
import time
import queue
import logging
import threading
import torch
import tensorrt as trt
import numpy as np
import base64
from pathlib import Path
from typing import Optional, Callable

logging.getLogger("ultralytics").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
UNET_ENGINE    = "unet_fp16.engine"
YOLO_ENGINE    = "marine_yolo26m_v1/weights/last.engine"
RESIZE         = (512, 512)
CONF_THRESHOLD = 0.65
TARGET_FPS     = 30          # cap output video fps
WRITER_Q_SZ    = 8
_DEVICE        = "cuda"
_DTYPE         = torch.float16

CLASS_NAMES = {
    0: "fish",       1: "coral",         2: "diver",      3: "turtle",
    4: "jellyfish",  5: "starfish",      6: "sea_cucumber", 7: "sea_urchin",
    8: "scallop",    9: "debris",        10: "plant",     11: "other_marine",
    12: "naval_mine"
}
CLASS_COLORS = {
    0:  (255, 180,  80),  1:  ( 80, 255, 180),  2:  ( 80, 180, 255),
    3:  (255, 255,  80),  4:  (200,  80, 255),  5:  ( 80, 255,  80),
    6:  (255, 140, 200),  7:  (140, 255, 255),  8:  (255, 200, 140),
    9:  (160, 160, 160),  10: (100, 220, 100),  11: (180, 180, 255),
    12: (  0,   0, 255),
}
MINE_CLS_ID = 12

# ─────────────────────────────────────────────
# TensorRT UNet wrapper
# ─────────────────────────────────────────────
class TRTUNet:
    def __init__(self, engine_path: str):
        self._logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self._logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self._logger) as rt:
            self.engine  = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.context.set_tensor_address("input",  int(x.data_ptr()))
        self.context.set_tensor_address("output", int(out.data_ptr()))
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle=stream)
        return out

# ─────────────────────────────────────────────
# Mine / object tracker
# ─────────────────────────────────────────────
class FastMineCounter:
    def __init__(self, persistence: int = 5):
        self.seen_mine_ids: set = set()
        self.persistence     = persistence
        self.active_tracks   = {}

    def reset(self):
        self.seen_mine_ids.clear()
        self.active_tracks.clear()

    def process(self, results):
        seen_this_frame: set = set()
        new_mines = 0
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            for i, box in enumerate(result.boxes):
                cls_id        = int(box.cls[0])
                conf          = float(box.conf[0])
                x1,y1,x2,y2  = map(int, box.xyxy[0])
                tid           = int(result.boxes.id[i])
                seen_this_frame.add(tid)
                self.active_tracks[tid] = {
                    "box": (x1,y1,x2,y2), "cls": cls_id,
                    "conf": conf, "ttl": self.persistence
                }
                if cls_id == MINE_CLS_ID and tid not in self.seen_mine_ids:
                    self.seen_mine_ids.add(tid)
                    new_mines += 1
        for tid in list(self.active_tracks):
            if tid not in seen_this_frame:
                self.active_tracks[tid]["ttl"] -= 1
                if self.active_tracks[tid]["ttl"] <= 0:
                    del self.active_tracks[tid]
        dets = [(t["cls"], t["conf"], *t["box"]) for t in self.active_tracks.values()]
        return dets, new_mines

# ─────────────────────────────────────────────
# Lazy model loader
# ─────────────────────────────────────────────
_unet        = None
_yolo_model  = None
_models_lock = threading.Lock()

def _load_models():
    from ultralytics import YOLO
    global _unet, _yolo_model
    with _models_lock:
        if _unet is None:
            logger.info("Loading UNet TRT engine…")
            _unet = TRTUNet(UNET_ENGINE)
            logger.info("✓ UNet loaded")
        if _yolo_model is None:
            logger.info("Loading YOLO TRT engine…")
            _yolo_model = YOLO(YOLO_ENGINE, task="detect")
            logger.info("✓ YOLO loaded")
            # Warmup
            dummy = torch.zeros(1, 3, *RESIZE, device=_DEVICE, dtype=_DTYPE)
            out   = _unet(dummy).clamp_(0, 1).contiguous()
            _yolo_model.track(out, conf=CONF_THRESHOLD, device=_DEVICE,
                              persist=False, half=True, verbose=False)
            logger.info("✓ Warmup complete")

# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────
def _bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(bgr).to(device=_DEVICE, non_blocking=True)
    t = t.permute(2, 0, 1).unsqueeze(0)[:, [2,1,0], :, :]
    return t.to(dtype=_DTYPE) / 255.0

def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    return (
        t.squeeze(0).permute(1,2,0).flip(-1)
        .cpu().float().clamp_(0,1).mul_(255).byte().numpy()
    )

def draw_detections(frame: np.ndarray, detections, total_mines: int) -> np.ndarray:
    mine_visible = any(d[0] == MINE_CLS_ID for d in detections)
    for cls_id, conf, x1, y1, x2, y2 in detections:
        color     = CLASS_COLORS.get(cls_id, (255,255,255))
        cls_name  = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        thickness = 3 if cls_id == MINE_CLS_ID else 2
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
        label          = f"{cls_name} {conf:.2f}"
        (tw,th), bl    = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        ty             = max(y1 - 2, th + 4)
        cv2.rectangle(frame, (x1, ty-th-bl-1), (x1+tw+3, ty+1), color, -1)
        cv2.putText(frame, label, (x1+1, ty-bl),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)
    if mine_visible:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (frame.shape[1], 40), (0,0,200), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"WARNING: NAVAL MINE DETECTED  (unique: {total_mines})",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return frame

def draw_hud(frame: np.ndarray, frame_idx: int, fps: float,
             total_mines: int, det_count: int) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-28), (w, h), (2,12,24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    txt = (f"Frame {frame_idx:05d}  |  {fps:.1f} FPS  |  "
           f"Det: {det_count}  |  Mines: {total_mines}  |  TENSORRT FP16")
    cv2.putText(frame, txt, (8, h-9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)
    return frame

# ─────────────────────────────────────────────
# Video processing
# ─────────────────────────────────────────────
def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Optional[Callable[[int,int], None]] = None,
    frame_callback: Optional[Callable] = None,
    yolo_enabled: bool = True,
    hud_enabled:  bool = True,
) -> dict:
    """
    Process video through UNet + optional YOLO pipeline.

    frame_callback(frame_idx, total_frames, jpeg_bytes, fps, det_count, mine_count)
        Called after every processed frame; use for live SSE streaming.
    progress_callback(current_frame, total_frames)
        Lightweight progress tick.
    """
    _load_models()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    src_fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    out_fps = min(src_fps, TARGET_FPS)

    # Writer thread
    write_q = queue.Queue(maxsize=WRITER_Q_SZ)
    class _Writer(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.vw = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                out_fps, RESIZE
            )
        def run(self):
            while True:
                item = write_q.get()
                if item is None:
                    break
                self.vw.write(item)
            self.vw.release()

    writer = _Writer()
    writer.start()

    counter     = FastMineCounter()
    counter.reset()
    total_mines = 0
    frame_idx   = 0
    t0          = time.time()

    cap     = cv2.VideoCapture(input_path)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=_DTYPE)

    with torch.no_grad(), amp_ctx:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            resized = cv2.resize(frame, RESIZE)
            tensor  = _bgr_to_tensor(resized)

            # UNet
            enhanced = _unet(tensor).clamp_(0, 1).contiguous()

            # YOLO
            dets = []
            if yolo_enabled:
                results = _yolo_model.track(
                    enhanced, conf=CONF_THRESHOLD, device=_DEVICE,
                    persist=True, tracker="bytetrack.yaml",
                    verbose=False, half=True
                )
                dets, new_mines = counter.process(results)
                total_mines += new_mines

            fps_now  = frame_idx / (time.time() - t0 + 1e-9)
            bgr      = _tensor_to_bgr(enhanced)
            annotated = bgr.copy()

            if yolo_enabled and dets:
                annotated = draw_detections(annotated, dets, total_mines)
            if hud_enabled:
                annotated = draw_hud(annotated, frame_idx, fps_now,
                                     total_mines, len(dets))

            write_q.put(annotated.copy())

            if frame_callback is not None:
                _, jpeg = cv2.imencode(".jpg", annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, 82])
                frame_callback(frame_idx, total_frames, jpeg.tobytes(),
                               fps_now, len(dets), total_mines)

            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)

    cap.release()
    write_q.put(None)
    writer.join()

    elapsed = time.time() - t0
    return {
        "output_file":         Path(output_path).name,
        "total_frames":        frame_idx,
        "elapsed_seconds":     round(elapsed, 2),
        "avg_fps":             round(frame_idx / (elapsed + 1e-9), 1),
        "total_mines_detected":total_mines,
        "yolo_enabled":        yolo_enabled,
        "hud_enabled":         hud_enabled,
    }

# ─────────────────────────────────────────────
# Image processing
# ─────────────────────────────────────────────
def process_image(
    input_path: str,
    output_path: str,
    yolo_enabled: bool = True,
) -> dict:
    """
    Process a single image.
    Returns original + enhanced as base64 JPEG strings, plus detection list.
    """
    _load_models()

    frame = cv2.imread(input_path)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {input_path}")

    resized = cv2.resize(frame, RESIZE)
    tensor  = _bgr_to_tensor(resized)

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=_DTYPE)
    with torch.no_grad(), amp_ctx:
        enhanced = _unet(tensor).clamp_(0, 1).contiguous()

        dets        = []
        total_mines = 0

        if yolo_enabled:
            results = _yolo_model.predict(
                enhanced, conf=CONF_THRESHOLD, device=_DEVICE,
                verbose=False, half=True
            )
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id      = int(box.cls[0])
                    conf        = float(box.conf[0])
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    dets.append((cls_id, conf, x1, y1, x2, y2))
                    if cls_id == MINE_CLS_ID:
                        total_mines += 1

        bgr       = _tensor_to_bgr(enhanced)
        annotated = bgr.copy()
        if yolo_enabled and dets:
            annotated = draw_detections(annotated, dets, total_mines)

    cv2.imwrite(output_path, annotated)

    _, orig_j = cv2.imencode(".jpg", resized,    [cv2.IMWRITE_JPEG_QUALITY, 90])
    _, enh_j  = cv2.imencode(".jpg", annotated,  [cv2.IMWRITE_JPEG_QUALITY, 90])

    return {
        "output_file":    Path(output_path).name,
        "detections":     len(dets),
        "mines_detected": total_mines,
        "original_b64":   base64.b64encode(orig_j.tobytes()).decode(),
        "enhanced_b64":   base64.b64encode(enh_j.tobytes()).decode(),
        "detection_list": [
            {
                "class":      CLASS_NAMES.get(d[0], f"cls_{d[0]}"),
                "confidence": round(d[1], 3),
                "bbox":       [d[2], d[3], d[4], d[5]],
            }
            for d in dets
        ],
    }
