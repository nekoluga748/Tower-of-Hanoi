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
    args = parser.parse_args()

    window_name = "Franka On-Hand Camera"
    frame_size = args.width * args.height * 3
    window_created = False
    first_frame_received = False
    pending_frame = bytearray()
    selector = selectors.DefaultSelector()
    stdin_fd = sys.stdin.fileno()

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.resizeWindow(window_name, args.width, args.height)
        black_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        cv2.imshow(window_name, black_frame)
        cv2.waitKey(1)
        selector.register(stdin_fd, selectors.EVENT_READ)
        print(f"OpenCV camera window ready ({args.width}x{args.height}); press q to close.", file=sys.stderr)

        while True:
            if selector.select(timeout=1.0 / 60.0):
                chunk = os.read(stdin_fd, frame_size - len(pending_frame))
                if not chunk:
                    break
                pending_frame.extend(chunk)
                if len(pending_frame) == frame_size:
                    if not first_frame_received:
                        print("First on-hand RGB frame received.", file=sys.stderr, flush=True)
                        first_frame_received = True
                    rgb_frame = np.frombuffer(pending_frame, dtype=np.uint8).reshape(
                        args.height, args.width, 3
                    )
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(window_name, bgr_frame)
                    pending_frame = bytearray()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except cv2.error as error:
        print(f"Unable to open OpenCV camera window: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    finally:
        selector.close()
        if window_created:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()