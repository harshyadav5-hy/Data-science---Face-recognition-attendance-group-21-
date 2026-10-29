"""Synthetic faces: a 3-D head (nose sticks out) versus a flat photo, projected like a camera."""
import math
import numpy as np

# units: half the eye distance. x right, y down, z towards the camera.
HEAD = np.array([(-1, 0, 0), (1, 0, 0), (0, 1.4, 1.1), (-0.7, 2.2, 0.3), (0.7, 2.2, 0.3)], float)
PHOTO = HEAD * np.array([1, 1, 0])          # the same face printed on a flat sheet


def rot_y(pts, deg):
    a = math.radians(deg)
    x = pts[:, 0] * math.cos(a) + pts[:, 2] * math.sin(a)
    z = -pts[:, 0] * math.sin(a) + pts[:, 2] * math.cos(a)
    return np.stack([x, pts[:, 1], z], 1)


def make_face(pts3, yaw=0.0, roll=0.0, scale=50.0, centre=(640, 360), noise=0.0, rng=None):
    p = rot_y(pts3, yaw)[:, :2]
    a = math.radians(roll)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p = (p - [0, 1.1]) @ R.T + [0, 1.1]
    p = p * scale + np.array(centre, float)
    if noise and rng is not None:
        p = p + rng.normal(0, noise, p.shape)
    f = np.zeros(15, np.float32)
    f[:4] = (centre[0] - 1.6 * scale, centre[1] - 1.5 * scale, 3.2 * scale, 4.0 * scale)
    f[4:14] = p.reshape(-1)
    f[14] = 0.99
    return f


def yaw_sequence(fps=15, seconds=10):
    """A person doing the challenge: straight, turn to +25, back, to -25, back, straight."""
    keys = [(0, 0), (0.8, 0), (1.6, 25), (2.4, 25), (3.2, 0), (4.0, -25), (4.8, -25), (5.6, 0), (seconds, 0)]
    ts = np.arange(0, seconds, 1.0 / fps)
    return ts, np.interp(ts, [k[0] for k in keys], [k[1] for k in keys])
