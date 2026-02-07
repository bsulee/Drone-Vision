"""Generate synthetic test videos for testing.

Usage:
    python tests/fixtures/generate_test_videos.py [output_dir]

Creates reusable test video files that can be used outside pytest fixtures.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def generate_video(
    output_path: str,
    num_frames: int,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> str:
    """Generate a synthetic test video with frame numbers burned in.

    Args:
        output_path: Where to save the video file.
        num_frames: Total number of frames to generate.
        fps: Frames per second for the video.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        The output path as a string.
    """
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Burn in frame number for visual verification.
        cv2.putText(
            frame, f"Frame {i}", (50, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3,
        )
        # Add a timestamp line for additional context.
        timestamp_ms = (i / fps) * 1000
        cv2.putText(
            frame, f"{timestamp_ms:.0f}ms", (50, height // 2 + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2,
        )
        writer.write(frame)

    writer.release()
    return output_path


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
