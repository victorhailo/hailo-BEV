# pylint: disable=C0301
"""
Ground-truth boxes from the nuScenes annotations, shown in place of the Hailo detections (--ground-truth).
"""
import queue
import numpy as np
import pacing
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name

# Same detection range per class as the post-processing of the model's detections (lidar_nusc_box_to_global)
CLASS_RANGE = {'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50, 'construction_vehicle': 50,
               'pedestrian': 40, 'motorcycle': 40, 'bicycle': 40, 'traffic_cone': 30, 'barrier': 30}

def load_ground_truth(data_path, scenes, nusc_tiny) -> dict:
    """
    Loads the annotated boxes of every sample to display, in the same format as the model's detections.

    Only the boxes the model is trained to detect are kept: the 10 detection classes, objects with at least
    one lidar or radar point, and within the same range per class as the detections.
    The annotations are made at the keyframes (2 Hz); for frames between keyframes (--sweeps data), the boxes
    are interpolated between the two surrounding keyframes.

    Args:
        data_path (str): Path to the nuScenes dataset (the annotations are read from it).
        scenes (list): Scenes from tokens.json, each with its sample 'tokens'.
        nusc_tiny (dict): The prepared nusc_tiny_dataset.json, for the ego poses.

    Returns:
        dict: Sample token -> list of annotation dicts (translation, size, rotation, category_name, ...).
    """
    nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=False)
    ground_truth = {}
    for scene in scenes:
        for token in scene['tokens']:
            if token.startswith('first'):
                continue
            ego_position = np.array(nusc_tiny[token]['LIDAR_TOP'][0]['translation'][:2])
            # Data prepared before sample_data tokens were stored only has keyframes, keyed by sample token
            lidar_token = nusc_tiny[token]['LIDAR_TOP'][2].get('sample_data_token') or nusc.get('sample', token)['data']['LIDAR_TOP']
            annos = []
            for box in nusc.get_boxes(lidar_token):
                ann = nusc.get('sample_annotation', box.token)
                name = category_to_detection_name(ann['category_name'])
                if name is None or ann['num_lidar_pts'] + ann['num_radar_pts'] == 0:
                    continue
                if np.linalg.norm(box.center[:2] - ego_position) > CLASS_RANGE[name]:
                    continue
                annos.append({'sample_token': token, 'translation': box.center.tolist(), 'size': box.wlh.tolist(),
                              'rotation': box.orientation.elements.tolist(), 'category_name': ann['category_name'],
                              'detection_name': name, 'detection_score': 1.0})
            ground_truth[token] = annos
    return ground_truth

def ground_truth_proc(out_queue, scenes, ground_truth, demo, run_slow, realtime=False) -> None:
    """
    Sends the ground truth of each sample to the visualization, in the same order as the Hailo pipeline
    (which skips the first sample of each scene), until the termination signal is received.

    Args:
        out_queue (Queue): Queue to the visualization, receives (annos, token).
        scenes (list): Scenes from tokens.json, each with its sample 'tokens'.
        ground_truth (dict): Output of load_ground_truth().
        demo (object): A demo object that handles termination signals.
        run_slow (bool): If True, sends 5 samples per second at most.
        realtime (bool): If True, the samples are not sent faster than they were recorded (for the ~12 Hz sweeps).
    """
    pacer = pacing.FramePacer(5 if run_slow else None)
    while True:
        for scene in scenes:
            for i, token in enumerate(scene['tokens']):
                if token.startswith('first'):
                    continue
                pacer.wait(scene['timestamps'][i] if realtime else None)
                while not demo.get_terminate():
                    try:
                        out_queue.put((ground_truth[token], token), block=True, timeout=0.5)
                        break
                    except queue.Full:
                        pass
                if demo.get_terminate():
                    return
