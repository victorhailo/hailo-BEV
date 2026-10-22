"""
Pacing of the frames sent into the pipeline.
"""
import time

# Recorded gaps longer than this are not waited for (e.g. between two scenes)
MAX_RECORDED_GAP = 1.0

class FramePacer:
    """
    Waits between frames so that they are sent at most at max_fps (if given), and not faster than they were
    recorded (if their timestamps are given). It does not try to catch up when the pipeline is slower.
    """
    def __init__(self, max_fps=None):
        self.max_fps = max_fps
        self.last_time = None
        self.last_timestamp = None

    def wait(self, timestamp=None) -> None:
        """
        Waits until the next frame can be sent.

        Args:
            timestamp (float, optional): Capture time of the next frame in seconds, to play it at the recorded pace.
        """
        if self.last_time is not None:
            interval = 1 / self.max_fps if self.max_fps else 0
            if timestamp is not None and self.last_timestamp is not None:
                recorded_gap = timestamp - self.last_timestamp
                if 0 < recorded_gap < MAX_RECORDED_GAP:
                    interval = max(interval, recorded_gap)
            while time.time() - self.last_time < interval:
                time.sleep(0.002)
        self.last_time = time.time()
        self.last_timestamp = timestamp
