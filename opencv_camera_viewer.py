from __future__ import annotations

import argparse
import os
import selectors
import sys

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    args = parser.parse_args()

    window_name = "Franka On-Hand Camera"
    pixel_count = args.width * args.height
    rgb_size = pixel_count * 3
    depth_size = pixel_count * 2
    frame_size = rgb_size + depth_size
    window_created = False
    first_frame_received = False
    show_depth = False
    pending_frame = bytearray()
    selector = selectors.DefaultSelector()
    stdin_fd = sys.stdin.fileno()

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.resizeWindow(window_name, args.width, args.height)
        cv2.moveWindow(window_name, args.x, args.y)
        black_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        cv2.imshow(window_name, black_frame)
        cv2.waitKey(1)
        selector.register(stdin_fd, selectors.EVENT_READ)
        print(
            f"OpenCV RGB-D window ready ({args.width}x{args.height}); press d to toggle depth, q to close.",
            file=sys.stderr,
        )

        while True:
            if selector.select(timeout=1.0 / 60.0):
                chunk = os.read(stdin_fd, frame_size - len(pending_frame))
                if not chunk:
                    break
                pending_frame.extend(chunk)
                if len(pending_frame) == frame_size:
                    if not first_frame_received:
                        print("First on-hand RGB-D frame received.", file=sys.stderr, flush=True)
                        first_frame_received = True
                    if show_depth:
                        depth_mm = np.frombuffer(
                            pending_frame, dtype=np.uint16, count=pixel_count, offset=rgb_size
                        ).reshape(args.height, args.width)
                        depth_preview = np.clip((1500 - depth_mm) * (255.0 / 1350), 0, 255).astype(np.uint8)
                        depth_view = cv2.applyColorMap(depth_preview, cv2.COLORMAP_TURBO)
                        depth_view[depth_mm == 0] = 0
                        cv2.putText(
                            depth_view,
                            "DEPTH (mm) | D: RGB",
                            (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(window_name, depth_view)
                    else:
                        rgb_frame = np.frombuffer(
                            pending_frame, dtype=np.uint8, count=rgb_size
                        ).reshape(args.height, args.width, 3)
                        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            bgr_frame,
                            "RGB | D: depth",
                            (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(window_name, bgr_frame)
                    pending_frame = bytearray()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_depth = not show_depth
    except cv2.error as error:
        print(f"Unable to open OpenCV camera window: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    finally:
        selector.close()
        if window_created:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()