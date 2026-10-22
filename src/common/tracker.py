# pylint: disable=C0301
"""
3D multi-object tracker for the detections (--tracker), in the style of SimpleTrack / ByteTrackV2.

Detections are matched to tracks by the distance between their centers on the ground (in global coordinates, so
the ego motion does not matter), in two rounds as in ByteTrack: the detections above the score threshold first,
then the lower-score ones, which can only keep existing tracks alive. Each track has a constant-velocity Kalman
filter on its ground position, so it estimates its own velocity (the model's is not reliable) and smooths the
position. The size and height are smoothed, and a heading that flips by 180 degrees is folded back.

The default settings were tuned on the nuScenes-mini scenes against the annotations (tuned on 5 scenes, checked on
the 5 others).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

# Max distance between the predicted position of a track and a detection to match them, per class, in meters
MATCH_DISTANCE = {'car': 4.0, 'truck': 5.0, 'bus': 6.0, 'trailer': 6.0, 'construction_vehicle': 5.0,
                  'pedestrian': 2.0, 'motorcycle': 3.0, 'bicycle': 3.0, 'traffic_cone': 2.0, 'barrier': 2.0}
# Standard deviation of the acceleration (process noise of the Kalman filter), per class, in m/s^2
ACCELERATION_NOISE = {'car': 3.0, 'truck': 3.0, 'bus': 3.0, 'trailer': 3.0, 'construction_vehicle': 3.0,
                      'pedestrian': 1.5, 'motorcycle': 3.0, 'bicycle': 2.0, 'traffic_cone': 0.3, 'barrier': 0.3}
# Classes whose heading follows their direction of travel
MOVING_CLASSES = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle', 'pedestrian'}

def wrap_angle(angle):
    """Wraps an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def yaw_of(quaternion) -> float:
    """Yaw (rotation around z) of a (w, x, y, z) quaternion."""
    w, x, y, z = quaternion
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

class Track:
    """
    A tracked object: Kalman filter on [x, y, vx, vy], smoothed height, size and heading.
    """
    def __init__(self, track_id, detection, position_noise):
        self.id = track_id
        self.name = detection['detection_name']
        self.category_name = detection.get('category_name', '')
        x, y, z = detection['translation']
        self.state = np.array([x, y, 0.0, 0.0])
        # Unknown velocity at first
        self.covariance = np.diag([position_noise ** 2, position_noise ** 2, 10.0 ** 2, 10.0 ** 2])
        self.z = z
        self.size = np.array(detection['size'], dtype=float)
        self.yaw = yaw_of(detection['rotation'])
        self.score = detection['detection_score']
        self.hits = 1
        self.time_since_update = 0.0

    def predict(self, dt):
        """Moves the track forward by dt seconds."""
        transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        sigma = ACCELERATION_NOISE.get(self.name, 2.0)
        # Piecewise-constant white acceleration
        q = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * sigma ** 2
        noise = np.zeros((4, 4))
        noise[np.ix_([0, 2], [0, 2])] = q
        noise[np.ix_([1, 3], [1, 3])] = q
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + noise
        self.time_since_update += dt

    def update(self, detection, position_noise, smoothing):
        """Corrects the track with a matched detection."""
        observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        innovation = np.array(detection['translation'][:2]) - observation @ self.state
        innovation_covariance = observation @ self.covariance @ observation.T + np.eye(2) * position_noise ** 2
        # Closed-form 2x2 inverse: np.linalg.inv goes through LAPACK, whose OpenBLAS worker threads then busy-wait
        # between frames and burn a CPU core
        (s00, s01), (s10, s11) = innovation_covariance
        inverse = np.array([[s11, -s01], [-s10, s00]]) / (s00 * s11 - s01 * s10)
        gain = self.covariance @ observation.T @ inverse
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(4) - gain @ observation) @ self.covariance

        self.z += smoothing * (detection['translation'][2] - self.z)
        self.size += smoothing * (np.array(detection['size']) - self.size)
        # A heading flipped by 180 degrees is folded back before smoothing
        yaw = yaw_of(detection['rotation'])
        if abs(wrap_angle(yaw - self.yaw)) > np.pi / 2:
            yaw += np.pi
        self.yaw = wrap_angle(self.yaw + smoothing * wrap_angle(yaw - self.yaw))
        # Moving objects face their direction of travel: fixes a track started with a flipped heading
        speed = np.hypot(self.state[2], self.state[3])
        if self.name in MOVING_CLASSES and speed > 2.0 and self.hits >= 3:
            if abs(wrap_angle(self.yaw - np.arctan2(self.state[3], self.state[2]))) > np.pi / 2:
                self.yaw = wrap_angle(self.yaw + np.pi)

        self.score += smoothing * (detection['detection_score'] - self.score)
        self.category_name = detection.get('category_name', self.category_name)
        self.hits += 1
        self.time_since_update = 0.0

    def to_annotation(self, sample_token) -> dict:
        """The track in the same format as the detections."""
        return {'sample_token': sample_token, 'translation': [self.state[0], self.state[1], self.z],
                'size': self.size.tolist(), 'rotation': [np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)],
                'velocity': [self.state[2], self.state[3]], 'detection_name': self.name, 'detection_score': self.score,
                'category_name': self.category_name, 'tracking_id': str(self.id), 'token': str(self.id)}

class Tracker:
    """
    Tracks the detections of consecutive frames of a scene. Call reset() at the start of each scene.

    Args:
        score_threshold (float): Detections from this score start tracks and are matched first.
        low_score_threshold (float): Detections from this score (and below score_threshold) only keep tracks alive.
        min_confirm_time (float): A track is shown once it has been detected over this time, in seconds
            (3 frames at ~12 Hz, the first frame at 2 Hz).
        max_age (float): Time a track is kept (and shown, at its predicted position) without detections, in seconds.
        position_noise (float): Standard deviation of the detected positions, in meters.
        smoothing (float): Weight of a new detection in the smoothed height, size, heading and score (0-1).
        default_dt (float): Time between frames when no timestamps are given, in seconds.
    """
    def __init__(self, score_threshold, low_score_threshold=0.25, min_confirm_time=0.17, max_age=0.25,
                 position_noise=2.0, smoothing=0.3, default_dt=0.5):
        self.score_threshold = score_threshold
        self.low_score_threshold = low_score_threshold
        self.min_confirm_time = min_confirm_time
        self.max_age = max_age
        self.position_noise = position_noise
        self.smoothing = smoothing
        self.default_dt = default_dt
        self.reset()

    def reset(self) -> None:
        """Forgets every track, e.g. at the start of a scene."""
        self.tracks = []
        self.next_id = 0
        self.last_timestamp = None
        self.frame_interval = None
        self.min_hits = 1

    def _match(self, tracks, detections):
        """Matches tracks and detections of the same class by distance (Hungarian algorithm), within the class gate."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost = np.full((len(tracks), len(detections)), 1e6)
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                if track.name == detection['detection_name']:
                    distance = np.hypot(*(np.array(detection['translation'][:2]) - track.state[:2]))
                    if distance <= MATCH_DISTANCE.get(track.name, 2.0):
                        cost[i, j] = distance
        rows, cols = linear_sum_assignment(cost)
        matches = [(i, j) for i, j in zip(rows, cols) if cost[i, j] < 1e6]
        matched_tracks, matched_detections = {i for i, _ in matches}, {j for _, j in matches}
        return (matches, [i for i in range(len(tracks)) if i not in matched_tracks],
                [j for j in range(len(detections)) if j not in matched_detections])

    def update(self, detections, sample_token, timestamp=None) -> list:
        """
        Updates the tracks with the detections of a new frame.

        Args:
            detections (list): Detections of the frame (dicts with translation, size, rotation, detection_name,
                detection_score), in global coordinates, of any score.
            sample_token (str): Token of the frame, copied into the output.
            timestamp (float, optional): Capture time of the frame in seconds.

        Returns:
            list: The tracks to show, in the same format as the detections, with a 'tracking_id'.
        """
        if timestamp is not None and self.last_timestamp is not None and 0 < timestamp - self.last_timestamp < 2:
            dt = timestamp - self.last_timestamp
        else:
            dt = self.default_dt
        self.last_timestamp = timestamp
        # Detections needed to show a track, from the smoothed frame interval (the sweeps alternate 50 and 100 ms)
        self.frame_interval = dt if self.frame_interval is None else 0.8 * self.frame_interval + 0.2 * dt
        self.min_hits = 1 + round(self.min_confirm_time / self.frame_interval)
        for track in self.tracks:
            track.predict(dt)

        high = [d for d in detections if d['detection_score'] >= self.score_threshold]
        low = [d for d in detections if self.low_score_threshold <= d['detection_score'] < self.score_threshold]

        # First round: high-score detections, with every track
        matches, unmatched_tracks, unmatched_high = self._match(self.tracks, high)
        for i, j in matches:
            self.tracks[i].update(high[j], self.position_noise, self.smoothing)
        # Second round: low-score detections, only to keep the remaining confirmed tracks alive
        remaining = [self.tracks[i] for i in unmatched_tracks if self.tracks[i].hits >= self.min_hits]
        matches, _, _ = self._match(remaining, low)
        for i, j in matches:
            remaining[i].update(low[j], self.position_noise, self.smoothing)

        # A track not confirmed yet is dropped as soon as it misses a frame; a confirmed one after max_age
        self.tracks = [t for t in self.tracks if t.time_since_update == 0 or
                       (t.hits >= self.min_hits and t.time_since_update <= self.max_age)]
        # New tracks from the unmatched high-score detections, except next to an existing track of the same class:
        # those are duplicate detections of a tracked object, and would start duplicate tracks
        for j in unmatched_high:
            position = np.array(high[j]['translation'][:2])
            gate = MATCH_DISTANCE.get(high[j]['detection_name'], 2.0)
            if any(t.name == high[j]['detection_name'] and np.hypot(*(t.state[:2] - position)) <= gate for t in self.tracks):
                continue
            self.tracks.append(Track(self.next_id, high[j], self.position_noise))
            self.next_id += 1

        return [t.to_annotation(sample_token) for t in self.tracks if t.hits >= self.min_hits]
