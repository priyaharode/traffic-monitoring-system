"""
Intelligent Traffic Monitoring System - Vision Subsystem
=========================================================
Module: vision/detector.py
Functionality:
  1. Multi-Model Architecture:
     - YOLO11n + ByteTrack for vehicle tracking & virtual stop-line enforcement
     - Custom YOLOv8 (best.pt) for No-Helmet, Triple-Riding, and License Plate detection
  2. Spatial-Temporal Cooldown & Per-Track Deduplication to prevent database flooding
  3. Real-time in-memory photographic evidence crop extraction & JPEG encoding
  4. Closed-loop REST ingestion with FastAPI backend:
     - POST /api/violations (Metadata with vehicle_number=None, actual confidence)
     - POST /api/violations/{id}/evidence (Evidence image with SHA-256 fingerprinting)
  5. Real-time visual telemetry HUD & annotated video feed generation
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# Project filesystem layout
VISION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISION_DIR.parent
DEFAULT_VIDEO = PROJECT_ROOT / "dataset" / "videos" / "traffic_sample.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "videos" / "traffic_detected.mp4"
DEFAULT_TRACKING_MODEL = PROJECT_ROOT / "yolo11n.pt"
DEFAULT_CUSTOM_MODEL = VISION_DIR / "weights" / "best.pt"

# COCO vehicle class identifiers for stop-line tracking
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Mapping custom YOLOv8 model class names to backend API violation types
CUSTOM_VIOLATION_MAP = {
    "WithoutHelmet": "no_helmet",
    "TripleRiding": "triple_riding",
}


class TrafficEnforcementDetector:
    """Multi-model traffic detector: vehicle tracking, stop-line enforcement, and custom rider violation detection."""

    def __init__(
        self,
        tracking_model_path: Path = DEFAULT_TRACKING_MODEL,
        custom_model_path: Path = DEFAULT_CUSTOM_MODEL,
        api_base_url: str = "http://127.0.0.1:8000",
        camera_id: str = "CAM_01_NORTH",
        stop_line_y: int = 195,
        stop_line_x_max: int = 420,
        custom_conf_thresh: float = 0.40,
        enable_api: bool = True,
        cooldown_frames: int = 45,
        cooldown_dist_px: float = 80.0,
        require_twowheeler: bool = True,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.camera_id = camera_id
        self.stop_line_y = stop_line_y
        self.stop_line_x_max = stop_line_x_max
        self.custom_conf_thresh = custom_conf_thresh
        self.enable_api = enable_api
        self.cooldown_frames = cooldown_frames
        self.cooldown_dist_px = cooldown_dist_px
        self.require_twowheeler = require_twowheeler

        # Validate weights exist
        if not tracking_model_path.exists():
            raise FileNotFoundError(f"Tracking model weights not found at: {tracking_model_path}")
        if not custom_model_path.exists():
            raise FileNotFoundError(f"Custom model weights not found at: {custom_model_path}")

        # Initialize models
        print(f"[VISION] Loading tracking model: {tracking_model_path.name}")
        self.tracking_model = YOLO(str(tracking_model_path))

        print(f"[VISION] Loading custom enforcement model: {custom_model_path.name}")
        self.custom_model = YOLO(str(custom_model_path))

        print(f"[VISION] Custom model classes: {self.custom_model.names}")

        # Dynamic mapping from model class IDs to class names
        self.custom_names_by_id = self.custom_model.names

        # Tracking and deduplication state
        self.prev_positions: Dict[int, float] = {}          # track_id -> previous centroid Y
        self.reported_stopline_tracks: Set[int] = set()      # Stop-line per-track deduplication
        
        # Spatial-temporal cooldown for un-tracked custom detections:
        # List of records: {"frame": int, "cx": float, "cy": float, "v_type": str}
        self.recent_custom_reports: List[dict] = []

        # Audit and telemetry logging
        self.violations_log: List[dict] = []
        self.counts_by_type: Dict[str, int] = {
            "stop_line": 0,
            "no_helmet": 0,
            "triple_riding": 0,
        }
        self.api_failures: int = 0

        # Verify backend connectivity if API is enabled
        if self.enable_api:
            self._verify_backend_connection()

    def _verify_backend_connection(self) -> None:
        """Check if FastAPI backend is available."""
        try:
            resp = requests.get(f"{self.api_base_url}/health", timeout=3.0)
            if resp.status_code == 200:
                print(f"[BACKEND] Connected to FastAPI backend at {self.api_base_url} (Health: OK)")
            else:
                print(f"[BACKEND] Warning: Backend returned status {resp.status_code}. Auto-reporting may fail.")
        except Exception as err:
            print(f"[BACKEND] Warning: Unable to reach backend at {self.api_base_url}: {err}")
            print("[BACKEND] Detections will continue locally; API dispatch will be attempted with fallbacks.")

    def _is_custom_in_cooldown(self, cx: float, cy: float, v_type: str, current_frame: int) -> bool:
        """
        Documented Spatial-Temporal Cooldown:
        Prevents multiple consecutive frames of the same moving rider from flooding the database.
        Checks if a detection of the same violation type occurred within `cooldown_frames`
        and within `cooldown_dist_px` Euclidean distance.
        """
        # Purge entries older than cooldown_frames
        self.recent_custom_reports = [
            r for r in self.recent_custom_reports
            if (current_frame - r["frame"]) <= self.cooldown_frames
        ]

        for r in self.recent_custom_reports:
            if r["v_type"] == v_type:
                dist = math.hypot(cx - r["cx"], cy - r["cy"])
                if dist < self.cooldown_dist_px:
                    return True
        return False

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        return inter / float(area1 + area2 - inter + 1e-6)

    def _has_twowheeler_context(self, c_box: np.ndarray, twowheeler_boxes: list, plate_boxes: list, max_dist: float = 140.0) -> bool:
        """
        Check if a helmet or triple-riding detection is associated with a two-wheeler.
        Considers COCO motorcycles (class 3), bicycles (class 1), or custom Plate detections.
        If no two-wheeler or plate exists in proximity, the detection is likely a pedestrian or background artifact.
        """
        if not self.require_twowheeler:
            return True

        cx = (c_box[0] + c_box[2]) / 2
        cy = (c_box[1] + c_box[3]) / 2

        for tb in twowheeler_boxes:
            tx = (tb[0] + tb[2]) / 2
            ty = (tb[1] + tb[3]) / 2
            if math.hypot(cx - tx, cy - ty) < max_dist:
                return True

        for pb in plate_boxes:
            px = (pb[0] + pb[2]) / 2
            py = (pb[1] + pb[3]) / 2
            if math.hypot(cx - px, cy - py) < max_dist:
                return True

        return False

    def _report_violation(
        self,
        violation_type: str,
        confidence: float,
        crop_image: np.ndarray,
        frame_idx: int,
        details: str = "",
    ) -> Optional[dict]:
        """
        Closed-loop auto-reporting:
        1. POST /api/violations (Metadata with vehicle_number=None, actual confidence)
        2. POST /api/violations/{id}/evidence (Evidence image upload & SHA-256 fingerprinting)
        """
        if not self.enable_api:
            return None

        # Format raw model confidence without inflation
        reported_conf = round(float(confidence), 2)

        # Stage 1: Ingest violation metadata (vehicle_number is None until OCR is integrated)
        create_payload = {
            "vehicle_number": None,
            "violation_type": violation_type,
            "camera_id": self.camera_id,
            "confidence": reported_conf,
        }

        try:
            create_resp = requests.post(
                f"{self.api_base_url}/api/violations",
                json=create_payload,
                timeout=15.0,
            )
            if create_resp.status_code != 201:
                self.api_failures += 1
                print(f"  [API ERROR] Failed to create violation: {create_resp.text}")
                return None

            violation_record = create_resp.json()
            violation_id = violation_record["id"]

            # Stage 2: Encode crop image to JPEG buffer
            success, jpeg_buffer = cv2.imencode(".jpg", crop_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                print(f"  [API ERROR] Could not encode evidence crop for violation #{violation_id}")
                return violation_record

            # Upload photographic evidence
            filename = f"evidence_{violation_type}_f{frame_idx}_id{violation_id}.jpg"
            files = {
                "file": (filename, jpeg_buffer.tobytes(), "image/jpeg")
            }
            upload_resp = requests.post(
                f"{self.api_base_url}/api/violations/{violation_id}/evidence",
                files=files,
                timeout=15.0,
            )

            if upload_resp.status_code == 200:
                evidence_data = upload_resp.json()
                sha_hash = evidence_data.get("evidence_hash", "N/A")
                short_hash = sha_hash[:16] if sha_hash else "None"
                print(f"  [DISPATCH] Violation #{violation_id} logged | Type: {violation_type} | "
                      f"Conf: {reported_conf*100:.0f}% | Details: {details} | SHA-256: {short_hash}...")
                return {
                    "id": violation_id,
                    "evidence_hash": sha_hash,
                }
            else:
                self.api_failures += 1
                print(f"  [API WARNING] Evidence upload returned {upload_resp.status_code}: {upload_resp.text}")
                return violation_record

        except Exception as exc:
            self.api_failures += 1
            print(f"  [API EXCEPTION] Error auto-reporting violation: {exc}")
            return None

    def process_video(
        self,
        video_path: Path = DEFAULT_VIDEO,
        output_path: Path = DEFAULT_OUTPUT,
        max_frames: Optional[int] = None,
    ) -> None:
        """Process video stream with multi-model inference, tracking, and evidence dispatch."""
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video stream: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open video writer for {output_path}")

        print("\n=======================================================")
        print("  RTO INTELLIGENT TRAFFIC ENFORCEMENT - MULTI-MODEL")
        print("=======================================================")
        print(f"  Input:            {video_path}")
        print(f"  Output:           {output_path}")
        print(f"  Resolution:       {width}x{height} @ {fps:.1f} FPS")
        print(f"  Total Frames:     {total_frames}")
        print(f"  Stop-Line:        Y = {self.stop_line_y} px (X: 0 -> {self.stop_line_x_max} px)")
        print(f"  Custom Conf Cut:  {self.custom_conf_thresh:.2f}")
        print(f"  API Pipeline:     {'ENABLED -> ' + self.api_base_url if self.enable_api else 'DISABLED (--no-api)'}")
        print("=======================================================\n")

        frame_idx = 0
        active_flash_frames = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    print(f"Reached max frame limit ({max_frames} frames). Finalizing video...")
                    break
                canvas = frame.copy()

                # -------------------------------------------------------------
                # 1. TRACKING MODEL: Vehicle tracking & Stop-Line Enforcement
                # -------------------------------------------------------------
                track_results = self.tracking_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )[0]

                active_tracks_in_frame = 0
                twowheeler_boxes = []

                if track_results.boxes is not None:
                    raw_t_boxes = track_results.boxes.xyxy.cpu().numpy()
                    raw_t_cls = track_results.boxes.cls.int().cpu().numpy()
                    # Class 1: bicycle, Class 3: motorcycle in COCO
                    twowheeler_boxes = [b for b, c in zip(raw_t_boxes, raw_t_cls) if c in (1, 3)]

                if track_results.boxes is not None and track_results.boxes.id is not None:
                    boxes = track_results.boxes.xyxy.cpu().numpy()
                    track_ids = track_results.boxes.id.int().cpu().numpy()
                    cls_ids = track_results.boxes.cls.int().cpu().numpy()
                    confs = track_results.boxes.conf.cpu().numpy()

                    for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
                        if cls_id not in VEHICLE_CLASSES:
                            continue

                        active_tracks_in_frame += 1
                        v_name = VEHICLE_CLASSES[cls_id]
                        x1, y1, x2, y2 = map(int, box)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # Check stop-line crossing (centroid Y crossed stop_line_y moving upward)
                        prev_y = self.prev_positions.get(track_id)
                        self.prev_positions[track_id] = float(cy)

                        is_crossing = (
                            prev_y is not None
                            and prev_y >= self.stop_line_y
                            and cy < self.stop_line_y
                            and cx <= self.stop_line_x_max
                        )

                        if is_crossing and track_id not in self.reported_stopline_tracks:
                            self.reported_stopline_tracks.add(track_id)
                            self.counts_by_type["stop_line"] += 1
                            active_flash_frames = 15

                            # Safety clamp crop coordinates
                            crop_y1 = max(0, y1 - 10)
                            crop_y2 = min(height, y2 + 10)
                            crop_x1 = max(0, x1 - 10)
                            crop_x2 = min(width, x2 + 10)
                            vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                            raw_conf = round(float(conf), 2)
                            print(f"\n[!] STOP-LINE BREACH at Frame {frame_idx}:")
                            print(f"    Target: Track #{track_id} ({v_name}) | Conf: {raw_conf*100:.0f}%")
                            print(f"    Trajectory: Centroid Y crossed {prev_y:.1f} -> {cy:.1f}")

                            api_result = self._report_violation(
                                violation_type="stop_line",
                                confidence=raw_conf,
                                crop_image=vehicle_crop,
                                frame_idx=frame_idx,
                                details=f"Track #{track_id} ({v_name})",
                            )

                            self.violations_log.append({
                                "frame": frame_idx,
                                "type": "stop_line",
                                "target": f"Track #{track_id} ({v_name})",
                                "confidence": raw_conf,
                                "api_id": api_result.get("id") if api_result else None,
                                "evidence_hash": api_result.get("evidence_hash") if api_result else None,
                            })

                        # Visual annotation for tracked vehicle
                        is_stopline_violator = track_id in self.reported_stopline_tracks
                        if is_stopline_violator:
                            box_color = (40, 40, 220)       # Crimson Red
                            label = f"STOP LINE INFRACTION #{track_id} ({v_name.upper()})"
                        else:
                            box_color = (160, 160, 160)     # Neutral Slate Gray
                            label = f"#{track_id} {v_name.upper()} {conf*100:.0f}%"

                        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
                        cv2.circle(canvas, (cx, cy), 3, (0, 255, 255), -1)

                        (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                        cv2.rectangle(canvas, (x1, y1 - 16), (x1 + lbl_w + 4, y1), box_color, -1)
                        cv2.putText(canvas, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

                # -------------------------------------------------------------
                # 2. CUSTOM MODEL: Helmet, Triple-Riding, & License Plate Scan
                # -------------------------------------------------------------
                custom_results = self.custom_model(
                    frame,
                    conf=self.custom_conf_thresh,
                    verbose=False,
                )[0]

                if custom_results.boxes is not None:
                    c_boxes = custom_results.boxes.xyxy.cpu().numpy()
                    c_cls_ids = custom_results.boxes.cls.int().cpu().numpy()
                    c_confs = custom_results.boxes.conf.cpu().numpy()

                    # Extract plate boxes in this frame for two-wheeler context association
                    plate_boxes = [
                        b for b, cid in zip(c_boxes, c_cls_ids)
                        if self.custom_names_by_id.get(cid, "").lower() == "plate"
                    ]

                    # Extract WithHelmet boxes for conflict resolution against false WithoutHelmet alarms
                    with_helmet_boxes = [
                        (b, float(conf))
                        for b, cid, conf in zip(c_boxes, c_cls_ids, c_confs)
                        if self.custom_names_by_id.get(cid, "").lower() == "withhelmet"
                    ]

                    for c_box, c_cls_id, c_conf in zip(c_boxes, c_cls_ids, c_confs):
                        c_name = self.custom_names_by_id.get(c_cls_id, f"class_{c_cls_id}")
                        cx1, cy1, cx2, cy2 = map(int, c_box)
                        center_x = (cx1 + cx2) / 2
                        center_y = (cy1 + cy2) / 2
                        raw_conf = round(float(c_conf), 2)

                        # A. Plate detection (informative only, not a violation)
                        if c_name.lower() == "plate":
                            plate_color = (255, 200, 0)  # Cyan / Light Blue
                            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), plate_color, 2)
                            p_lbl = f"PLATE {raw_conf*100:.0f}%"
                            cv2.putText(canvas, p_lbl, (cx1, max(12, cy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, plate_color, 1, cv2.LINE_AA)

                        # B. WithHelmet (compliant, not a violation)
                        elif c_name.lower() == "withhelmet":
                            helmet_color = (50, 180, 50)  # Forest Green
                            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), helmet_color, 2)
                            h_lbl = f"HELMET OK {raw_conf*100:.0f}%"
                            cv2.putText(canvas, h_lbl, (cx1, max(12, cy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, helmet_color, 1, cv2.LINE_AA)

                        # C. Infractions: WithoutHelmet or TripleRiding
                        elif c_name in CUSTOM_VIOLATION_MAP:
                            mapped_violation = CUSTOM_VIOLATION_MAP[c_name]

                            # Conflict Resolution: If this WithoutHelmet box overlaps with a WithHelmet detection,
                            # suppress the false positive on compliant riders wearing helmets
                            if mapped_violation == "no_helmet":
                                is_wearing_helmet = any(
                                    self._compute_iou(c_box, wb) > 0.35 and wc >= (raw_conf * 0.80)
                                    for wb, wc in with_helmet_boxes
                                )
                                if is_wearing_helmet:
                                    continue

                            # Contextual Guardrail: Verify proximity to a two-wheeler or plate
                            if not self._has_twowheeler_context(c_box, twowheeler_boxes, plate_boxes):
                                # Suppress false positive on isolated pedestrians/cyclists
                                continue

                            # Determine visual style
                            if mapped_violation == "no_helmet":
                                viol_color = (30, 30, 230)      # Vivid Crimson Red
                                viol_label = f"NO HELMET VIOLATION {raw_conf*100:.0f}%"
                            else:  # triple_riding
                                viol_color = (200, 30, 200)     # Magenta / Violet
                                viol_label = f"TRIPLE RIDING VIOLATION {raw_conf*100:.0f}%"

                            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), viol_color, 2)
                            (v_w, v_h), _ = cv2.getTextSize(viol_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                            cv2.rectangle(canvas, (cx1, max(0, cy1 - 16)), (cx1 + v_w + 4, cy1), viol_color, -1)
                            cv2.putText(canvas, viol_label, (cx1 + 2, max(11, cy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

                            # Apply spatial-temporal deduplication cooldown
                            in_cooldown = self._is_custom_in_cooldown(center_x, center_y, mapped_violation, frame_idx)

                            if not in_cooldown:
                                self.recent_custom_reports.append({
                                    "frame": frame_idx,
                                    "cx": center_x,
                                    "cy": center_y,
                                    "v_type": mapped_violation,
                                })
                                self.counts_by_type[mapped_violation] += 1

                                # Extract evidence crop around the offending bounding box
                                pad = 12
                                cr_y1 = max(0, cy1 - pad)
                                cr_y2 = min(height, cy2 + pad)
                                cr_x1 = max(0, cx1 - pad)
                                cr_x2 = min(width, cx2 + pad)
                                evidence_crop = frame[cr_y1:cr_y2, cr_x1:cr_x2]

                                print(f"\n[!] {c_name.upper()} DETECTED at Frame {frame_idx}:")
                                print(f"    Confidence: {raw_conf*100:.0f}% | Position: ({center_x:.0f}, {center_y:.0f})")

                                api_result = self._report_violation(
                                    violation_type=mapped_violation,
                                    confidence=raw_conf,
                                    crop_image=evidence_crop,
                                    frame_idx=frame_idx,
                                    details=f"{c_name} at ({center_x:.0f}, {center_y:.0f})",
                                )

                                self.violations_log.append({
                                    "frame": frame_idx,
                                    "type": mapped_violation,
                                    "target": f"{c_name} @ ({center_x:.0f}, {center_y:.0f})",
                                    "confidence": raw_conf,
                                    "api_id": api_result.get("id") if api_result else None,
                                    "evidence_hash": api_result.get("evidence_hash") if api_result else None,
                                })

                # -------------------------------------------------------------
                # 3. Visual Overlays & HUD
                # -------------------------------------------------------------
                # Stop-line visual overlay
                if active_flash_frames > 0:
                    line_color = (40, 40, 240)  # Flashing Crimson
                    line_thickness = 3
                    active_flash_frames -= 1
                    line_label = "[!] STOP LINE BREACH DETECTED"
                else:
                    line_color = (0, 215, 255)  # Amber Yellow
                    line_thickness = 2
                    line_label = "RTO ENFORCEMENT STOP-LINE (VIRTUAL)"

                cv2.line(canvas, (0, self.stop_line_y), (self.stop_line_x_max, self.stop_line_y), line_color, line_thickness)
                cv2.putText(canvas, line_label, (10, self.stop_line_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, line_color, 1, cv2.LINE_AA)

                # Heads-Up Display (HUD) top telemetry bar
                hud_overlay = canvas.copy()
                cv2.rectangle(hud_overlay, (0, 0), (width, 36), (15, 23, 42), -1)
                cv2.addWeighted(hud_overlay, 0.82, canvas, 0.18, 0, canvas)

                total_logged = sum(self.counts_by_type.values())
                hud_text = (
                    f"RTO ENFORCEMENT | {self.camera_id} | "
                    f"STOP-LINE: {self.counts_by_type['stop_line']} | "
                    f"NO-HELMET: {self.counts_by_type['no_helmet']} | "
                    f"TRIPLE-RIDE: {self.counts_by_type['triple_riding']} | "
                    f"TOTAL: {total_logged}"
                )
                cv2.putText(canvas, hud_text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                # Write annotated frame
                writer.write(canvas)

                if frame_idx % 60 == 0 or frame_idx == total_frames:
                    pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Processing: frame {frame_idx}/{total_frames} ({pct:.1f}%) | "
                          f"Infractions: {self.counts_by_type}")

        finally:
            cap.release()
            writer.release()

        # -------------------------------------------------------------
        # 4. End-of-Run Summary Report
        # -------------------------------------------------------------
        print("\n=======================================================")
        print("  PROCESSING COMPLETED SUCCESSFULLY")
        print("=======================================================")
        print(f"  Annotated video saved: {output_path}")
        print("  Infraction counts by type:")
        print(f"    - Stop-Line Breaches:  {self.counts_by_type['stop_line']}")
        print(f"    - No-Helmet Incidents: {self.counts_by_type['no_helmet']}")
        print(f"    - Triple-Riding Cases: {self.counts_by_type['triple_riding']}")
        print(f"    - Total Logged Events: {len(self.violations_log)}")
        if self.enable_api:
            print(f"    - API Request Failures: {self.api_failures}")
        print("\n  Detailed Incident Log:")
        if not self.violations_log:
            print("    (No violations detected)")
        else:
            for item in self.violations_log:
                api_str = f"Backend ID #{item['api_id']}" if item["api_id"] else "API: offline"
                hash_str = f"SHA-256: {item['evidence_hash'][:16]}..." if item.get("evidence_hash") else ""
                print(f"    - Frame {item['frame']:3d} | {item['type']:14s} | Conf: {item['confidence']*100:.0f}% | "
                      f"{item['target']:25s} | {api_str} {hash_str}")
        print("=======================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Traffic Vision Enforcement Pipeline")
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO), help="Path to input video")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to output video")
    parser.add_argument("--tracking-model", type=str, default=str(DEFAULT_TRACKING_MODEL), help="Path to tracking model (YOLO11)")
    parser.add_argument("--custom-model", type=str, default=str(DEFAULT_CUSTOM_MODEL), help="Path to custom model (best.pt)")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="FastAPI backend URL")
    parser.add_argument("--camera-id", type=str, default="CAM_01_NORTH", help="Camera identifier")
    parser.add_argument("--line-y", type=int, default=195, help="Y-coordinate for virtual stop line")
    parser.add_argument("--line-x-max", type=int, default=420, help="Maximum X-coordinate for stop line")
    parser.add_argument("--custom-conf", type=float, default=0.22, help="Confidence threshold for custom model")
    parser.add_argument("--cooldown-frames", type=int, default=45, help="Frame cooldown for custom detections")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--no-require-twowheeler", action="store_true", help="Disable two-wheeler proximity filter")
    parser.add_argument("--no-api", action="store_true", help="Disable API auto-reporting")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = TrafficEnforcementDetector(
        tracking_model_path=Path(args.tracking_model),
        custom_model_path=Path(args.custom_model),
        api_base_url=args.api_url,
        camera_id=args.camera_id,
        stop_line_y=args.line_y,
        stop_line_x_max=args.line_x_max,
        custom_conf_thresh=args.custom_conf,
        enable_api=not args.no_api,
        cooldown_frames=args.cooldown_frames,
        require_twowheeler=not args.no_require_twowheeler,
    )
    detector.process_video(
        video_path=Path(args.video),
        output_path=Path(args.output),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
