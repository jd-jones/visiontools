import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def genFrames(video_fn):
    cap = cv2.VideoCapture(video_fn)

    if not cap.isOpened():
        raise FileNotFoundError()

    frame_status, frame = cap.read()
    while(frame_status):
        yield frame
        frame_status, frame = cap.read()

    cap.release()


def readVideo(video_fn):
    frames = genFrames(video_fn)
    frames = np.stack(tuple(frames), axis=0)
    return frames


def writeVideo(frames, video_fn, frame_rate=30):
    frame_shape = tuple(int(i) for i in reversed(frames[0].shape[:2]))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_fn, fourcc, frame_rate, frame_shape, True)

    for frame in frames:
        out.write(frame)

    out.release()
