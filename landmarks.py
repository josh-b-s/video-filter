"""Pose-landmark topology and coordinate extraction: which landmark
indices/connections exist, converting MediaPipe's raw world-landmark
coordinates into this project's scene convention, visibility filtering,
and the synthetic neck/head marker used in face-free mode."""

from mediapipe.tasks.python.vision import PoseLandmarksConnections

import config

# MediaPipe's 33 pose landmarks, indexed 0-32.
FACE_LANDMARKS = set(range(0, 11))

_RAW_CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]

ALL_CONNECTIONS = _RAW_CONNECTIONS
ALL_LANDMARKS = sorted({i for pair in ALL_CONNECTIONS for i in pair})

BODY_CONNECTIONS = [
    (a, b) for (a, b) in _RAW_CONNECTIONS
    if a not in FACE_LANDMARKS and b not in FACE_LANDMARKS
]
BODY_LANDMARKS = sorted({i for pair in BODY_CONNECTIONS for i in pair})

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Synthetic head marker placed above the shoulder midpoint (meters),
# used only when face landmarks are excluded and there's nothing else
# to anchor a head to.
HEAD_OFFSET_UP = 0.28


def to_scene_coords(lm) -> tuple[float, float, float]:
    """MediaPipe world space: x right, y down, z toward camera. This
    project's convention swaps to (x, -z, -y), where slot 1 is
    depth-toward-camera and slot 2 is vertical-up -- see
    viewer_template.html's pointOf(), which expects that layout."""
    return (lm.x, -lm.z, -lm.y)


def extract_points(
    result,
    active_landmarks: list[int],
    smoother=None,
) -> dict[int, tuple[float, float, float]]:
    """Builds {landmark_index: (x, y, z)} for one frame's detection
    result, filtering out low-visibility landmarks and optionally
    smoothing each point across frames via `smoother`."""
    points: dict[int, tuple[float, float, float]] = {}
    if not result.pose_world_landmarks:
        return points

    frame_landmarks = result.pose_world_landmarks[0]
    for idx in active_landmarks:
        lm = frame_landmarks[idx]
        visibility = getattr(lm, "visibility", 1.0)
        if visibility is not None and visibility < config.min_landmark_visibility:
            continue
        x, y, z = to_scene_coords(lm)
        points[idx] = smoother.smooth(idx, x, y, z) if smoother else (x, y, z)

    return points


def add_synthetic_head(points: dict) -> None:
    """Mutates `points` in place, adding 'neck'/'head' keys derived from
    the shoulder midpoint. Only meaningful when face landmarks (0-10)
    were excluded from `points` to begin with."""
    if LEFT_SHOULDER not in points or RIGHT_SHOULDER not in points:
        return
    lx, ly, lz = points[LEFT_SHOULDER]
    rx, ry, rz = points[RIGHT_SHOULDER]
    neck = ((lx + rx) / 2, (ly + ry) / 2, (lz + rz) / 2)
    head = (neck[0], neck[1], neck[2] + HEAD_OFFSET_UP)
    points["neck"] = neck
    points["head"] = head
