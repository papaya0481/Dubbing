"""Detect faces in CHEM videos using RetinaFace with batch processing."""

import argparse
import cv2
from pathlib import Path
from batch_face import RetinaFace
from tqdm import tqdm
from multiprocessing import Pool

# Global detector for each worker process
_detector = None


def init_worker(gpu_id):
    """Initialize detector once per worker."""
    global _detector
    _detector = RetinaFace(gpu_id=gpu_id)


def resize_frame(frame, max_size=512):
    """Resize frame to max_size while keeping aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


def apply_sliding_window(detections, window_size=7):
    """Apply sliding window smoothing. If 2/3+ frames in window have faces, consider all as detected."""
    if len(detections) == 0:
        return detections

    smoothed = [False] * len(detections)
    threshold = window_size * 2 // 3

    for i in range(len(detections)):
        left = max(0, i - 3)
        right = min(len(detections), i + 4)
        window = detections[left:right]
        if sum(window) >= threshold:
            smoothed[i] = True

    return smoothed


def detect_faces_in_video(video_path, detector, threshold=0.95, batch_size=64):
    """Detect faces in all frames using batch processing."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, False, False, False

    detections = []
    frames_batch = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_frame(frame, max_size=512)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_batch.append(frame_rgb)

        if len(frames_batch) >= batch_size:
            for f in frames_batch:
                faces = detector(f, threshold=threshold, resize=1, max_size=-1, return_dict=True)
                detections.append(len(faces) > 0)
            frames_batch = []

    if frames_batch:
        for f in frames_batch:
            faces = detector(f, threshold=threshold, resize=1, max_size=-1, return_dict=True)
            detections.append(len(faces) > 0)

    cap.release()
    if len(detections) == 0:
        return False, False, False, False

    detected_count = sum(detections)
    has_detection = detected_count > 0
    all_detected = detected_count == len(detections)

    smoothed = apply_sliding_window(detections)
    smoothed_count = sum(smoothed)
    has_detection_smoothed = smoothed_count > 0
    all_detected_smoothed = smoothed_count == len(smoothed)

    return has_detection, all_detected, has_detection_smoothed, all_detected_smoothed


def process_video(args):
    """Worker function for multiprocessing."""
    video_path, threshold, batch_size = args
    has_det, all_det, has_det_smooth, all_det_smooth = detect_faces_in_video(
        video_path, _detector, threshold, batch_size
    )
    return not all_det, not has_det, not all_det_smooth, not has_det_smooth


def main():
    parser = argparse.ArgumentParser(description="Detect faces in CHEM videos")
    parser.add_argument("--input-dir", default="/data2/ruixin/downloads/chem_raw_processed1/videos")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    video_files = []
    for folder in sorted(Path(args.input_dir).iterdir()):
        if folder.is_dir():
            clipped_dir = folder / "clipped"
            if clipped_dir.exists():
                video_files.extend(sorted(clipped_dir.glob("*.mp4")))

    print(f"Found {len(video_files)} videos, using {args.workers} workers, batch_size={args.batch_size}")

    tasks = [(vf, args.threshold, args.batch_size) for vf in video_files]

    with Pool(processes=args.workers, initializer=init_worker, initargs=(args.gpu_id,)) as pool:
        results = list(tqdm(pool.imap(process_video, tasks), total=len(tasks), desc="Processing"))

    at_least_one_missing = sum(r[0] for r in results)
    all_frames_missing = sum(r[1] for r in results)
    at_least_one_missing_smooth = sum(r[2] for r in results)
    all_frames_missing_smooth = sum(r[3] for r in results)
    total = len(video_files)

    print(f"\n[Original]")
    print(f"At least one frame without face: {at_least_one_missing}/{total}")
    print(f"All frames without face: {all_frames_missing}/{total}")
    print(f"\n[With sliding window smoothing]")
    print(f"At least one frame without face: {at_least_one_missing_smooth}/{total}")
    print(f"All frames without face: {all_frames_missing_smooth}/{total}")


if __name__ == "__main__":
    main()
