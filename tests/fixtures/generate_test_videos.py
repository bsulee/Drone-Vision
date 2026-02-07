"""Generate synthetic test videos for testing.

Usage:
    python tests/fixtures/generate_test_videos.py [output_dir]

Creates reusable test video files that can be used outside pytest fixtures.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


_CODEC_ATTEMPTS = [
    ("avc1", ".mp4"),
    ("mp4v", ".mp4"),
    ("MJPG", ".avi"),
]


def generate_video(
    output_path: str,
    num_frames: int,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> str:
    """Generate a synthetic test video with frame numbers burned in.

    Tries multiple codecs for cross-platform compatibility.

    Args:
        output_path: Where to save the video file.
        num_frames: Total number of frames to generate.
        fps: Frames per second for the video.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        The actual output path (extension may differ if codec fallback used).
    """
    for codec, ext in _CODEC_ATTEMPTS:
        actual_path = output_path.rsplit(".", 1)[0] + ext
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(actual_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            continue

        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                frame, f"Frame {i}", (50, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3,
            )
            timestamp_ms = (i / fps) * 1000
            cv2.putText(
                frame, f"{timestamp_ms:.0f}ms", (50, height // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2,
            )
            writer.write(frame)

        writer.release()
        return actual_path

    raise RuntimeError(f"No working video codec found for {output_path}")


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures")
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        ("sample_10s_30fps.mp4", 300, 30.0),
        ("short_2s_30fps.mp4", 60, 30.0),
        ("short_1s_15fps.mp4", 15, 15.0),
    ]

    for filename, frames, fps in videos:
        path = str(output_dir / filename)
        generate_video(path, frames, fps)
        print(f"Generated: {path} ({frames} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
