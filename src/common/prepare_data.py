#!/usr/bin/env python3
import numpy as np
from numpy.lib.format import open_memmap
from nuscenes.nuscenes import NuScenes
import argparse
import json
import os
import sys
import cv2
sys.path.append('src/common')
import pre_post_process
import visualization

CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="BEV demo")
    parser.add_argument("-d", "--data", default="resources/data/", help="Path to the data folder, where the nuScenes dataset is.")
    parser.add_argument("-i", "--input", default="resources/input/", help="Path to the input folder. Use this flag only if you dont want to use the default input folder location.")
    parser.add_argument("-r", "--raw-data", action="store_true", default="False", help="Use this flag only if you want to run the demo from raw date input" )
    parser.add_argument("--sweeps", action="store_true", help="Use every camera frame (~12 Hz, including the unannotated sweeps between keyframes) instead of the annotated keyframes only (2 Hz).")
    parser.add_argument("-s", "--scenes", nargs="+", default=None, help="Scenes to prepare, in playback order, by index (0-9 in v1.0-mini) or name (e.g. scene-0103). Default: all scenes.")
    parsed_args = parser.parse_args()
    return parsed_args

def resolve_scenes(requested, nusc):
    """
    Converts scene indices or names into scene indices. Returns all scenes if nothing is requested.
    """
    names = [scene['name'] for scene in nusc.scene]
    if not requested:
        return list(range(len(names)))
    indices = []
    for scene in requested:
        if scene.isdigit() and int(scene) < len(names):
            indices.append(int(scene))
        elif scene in names:
            indices.append(names.index(scene))
        else:
            sys.exit(f'Unknown scene "{scene}". Available: 0-{len(names) - 1} or {", ".join(names)}')
    return indices

def keyframes(nusc, scene) -> list:
    """
    Frames of a scene at the annotated keyframes (2 Hz).

    Returns:
        list: (frame token, {channel: sample_data record}) per frame, the frame token being the sample token.
    """
    frames = []
    token = scene['first_sample_token']
    while token:
        sample = nusc.get('sample', token)
        frames.append((token, {channel: nusc.get('sample_data', sample['data'][channel]) for channel in CAMS + ['LIDAR_TOP']}))
        token = sample['next']
    return frames

def records_between(nusc, first_token, end_token) -> list:
    """
    sample_data records of a channel, from a record (included) until another one (excluded).
    """
    records = []
    token = first_token
    while token != end_token:
        records.append(nusc.get('sample_data', token))
        token = records[-1]['next']
    return records

def closest(records, timestamp) -> dict:
    """
    The record closest in time to a timestamp.
    """
    return min(records, key=lambda record: abs(record['timestamp'] - timestamp))

def sweep_frames(nusc, scene) -> list:
    """
    Frames of a scene at the camera rate (~12 Hz), including the unannotated sweeps between keyframes.

    The keyframes, where every channel has a record, are used as synchronization points. Between two keyframes,
    the cameras usually have the same number of images, taken in the same order, and are paired by position.
    When a camera has one image less, the image closest in time is used instead (so one of its images repeats).
    The LIDAR_TOP record (for the ego pose and the map) is the one closest in time to the CAM_FRONT image.
    Pairing by closest time only does not work well: the cameras' timestamps are on a 50 ms grid with
    different phases, and about a third of the images of some cameras would repeat.

    Returns:
        list: (frame token, {channel: sample_data record}) per frame, the frame token being the CAM_FRONT
        sample_data token.
    """
    frames = []
    sample = nusc.get('sample', scene['first_sample_token'])
    while sample['next']:
        following = nusc.get('sample', sample['next'])
        interval = {channel: records_between(nusc, sample['data'][channel], following['data'][channel])
                    for channel in CAMS + ['LIDAR_TOP']}
        lidar = interval['LIDAR_TOP'] + [nusc.get('sample_data', following['data']['LIDAR_TOP'])]
        for index, front in enumerate(interval['CAM_FRONT']):
            records = {'CAM_FRONT': front}
            for channel in CAMS[1:]:
                if len(interval[channel]) == len(interval['CAM_FRONT']):
                    records[channel] = interval[channel][index]
                else:
                    records[channel] = closest(interval[channel], front['timestamp'])
            records['LIDAR_TOP'] = closest(lidar, front['timestamp'])
            frames.append((front['token'], records))
        sample = following
    # The last keyframe
    frames.append((sample['data']['CAM_FRONT'], {channel: nusc.get('sample_data', sample['data'][channel])
                                                 for channel in CAMS + ['LIDAR_TOP']}))
    return frames

if __name__ == "__main__":
    args = parse_args()
    nusc = NuScenes(version='v1.0-mini', dataroot=args.data , verbose=False)
    data = {'sweeps': args.sweeps, 'scenes': []}
    if args.raw_data == True:
        print('Preparing data for running from raw data')
    else:
        print('Preparing data for running from jpg')

    scene_indices = resolve_scenes(args.scenes, nusc)
    os.makedirs(f'{args.input}/map_files', exist_ok=True)
    nusc_tiny_dataset = {}
    for scene_index in scene_indices:
        scene = nusc.scene[scene_index]
        frames = sweep_frames(nusc, scene) if args.sweeps else keyframes(nusc, scene)
        print(f'Preparing scene number {scene_index} ({scene["name"]}): {len(frames)} frames')
        # The first frame is listed twice: as 'first_<token>' it only provides the model's previous-frame input
        frames = [(f'first_{frames[0][0]}', frames[0][1])] + frames

        if args.raw_data == True:
            raw_file = f'{scene["token"]}{"_sweeps" if args.sweeps else ""}.npy'
            # Written straight to disk: with --sweeps, a scene is ~1 GB
            raw_input = open_memmap(f'{args.input}/{raw_file}', mode='w+', dtype=np.uint8,
                                    shape=(len(frames), len(CAMS), 320, 800, 3))
        tokens, timestamps = [], []
        for frame_index, (token, records) in enumerate(frames):
            nusc_tiny_dataset[token] = {}
            for channel in CAMS + ['LIDAR_TOP']:
                sd_record = records[channel]
                nusc_tiny_dataset[token][channel] = [nusc.get('ego_pose', sd_record['ego_pose_token']),
                                                     nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])]
                if channel != 'LIDAR_TOP':
                    nusc_tiny_dataset[token][channel].append({"filename": sd_record['filename'], "sample_data_token": sd_record['token']})
                else:
                    cv2.imwrite(f'{args.input}/map_files/{token}_LIDAR_TOP.jpg', visualization.render_ego_centric_map(nusc, sd_record['token']))
                    nusc_tiny_dataset[token][channel].append({"filename": f'map_files/{token}_LIDAR_TOP.jpg', "sample_data_token": sd_record['token']})
            if args.raw_data == True:
                raw_input[frame_index] = pre_post_process.preprocess(args.data, [records[cam]['filename'] for cam in CAMS])
            tokens.append(token)
            timestamps.append(records['CAM_FRONT']['timestamp'] / 1e6)

        scene_data = {'tokens': tokens, 'timestamps': timestamps}
        if args.raw_data == True:
            raw_input.flush()
            del raw_input
            scene_data['input'] = raw_file
        data['scenes'].append(scene_data)
    with open(f'{args.input}/tokens.json', "w") as json_file:
        json.dump(data, json_file, indent=4)
    with open(f'{args.input}/nusc_tiny_dataset.json', "w") as json_file:
        json.dump(nusc_tiny_dataset, json_file, indent=4)
