#!/usr/bin/env python3
# pylint: disable=C0301
# pylint: disable=E1101
# pylint: disable=C0411
# pylint: disable=W0621
# pylint: disable=C0103
# pylint: disable=E0401
# pylint: disable=C0413

import numpy as np
import threading
import time
import argparse
import queue
import json
import sys
import os
import signal
from hailo_platform import (Device, VDevice, HailoSchedulingAlgorithm)

sys.path.append('src/common')
import fps_calc
import multiprocessing
import pre_post_process
import core
import demo_manager
import async_api
import visualization
import ground_truth
import startup_checks

MAX_QUEUE_SIZE = 3

def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="BEV demo")
    parser.add_argument('--run-slow', action='store_true', help='Run the demo at 5 FPS for better visualization of 3D boxes.')
    parser.add_argument('--raw-data', action='store_true', help=' Run the demo from raw data for lower cpu usage.')
    parser.add_argument('--score-threshold', default=pre_post_process.DEFAULT_SCORE_THRESHOLD, type=float, help=f'Hide detections with a lower confidence score (default {pre_post_process.DEFAULT_SCORE_THRESHOLD}, 0 shows all).')
    parser.add_argument('--ground-truth', action='store_true', help='Show the nuScenes ground-truth boxes instead of the Hailo detections. Does not use the Hailo device, so it can run next to a Hailo instance for comparison.')
    parser.add_argument('--prev-frame-gap', default=core.DEFAULT_PREV_FRAME_GAP, type=float, help=f'Time between the current frame and the earlier frame the model uses with it, in seconds (default {core.DEFAULT_PREV_FRAME_GAP}). Mostly useful with --sweeps data.')
    parser.add_argument('--no-tracker', action='store_true', help='Show the detections of each frame as they are, without tracking the objects across frames.')
    parser.add_argument("-i", "--input", default="resources/input/", help="Path to the input folder, Use this flag only if you have modified the default input folder location.")
    parser.add_argument("-m", "--models", default="resources/models/", help="Path to the models folder, Use this flag only if you have modified the default models folder location.")
    parser.add_argument("-d", "--data", default="resources/data/", help="Path to the data folder, where the nuScenes dataset is.")

    parsed_args = parser.parse_args()
    return parsed_args

def run_ground_truth(args, nusc, scenes, sweeps) -> None:
    """
    Runs the demo with the nuScenes ground-truth boxes instead of the Hailo detections, without the Hailo device.
    """
    boxes = ground_truth.load_ground_truth(args.data, scenes, nusc)
    manager = multiprocessing.Manager()
    demo_mng = demo_manager.DemoManager(manager)
    viz_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)

    # Only the main process handles Ctrl-C, see the Hailo pipeline below
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    viz_process = multiprocessing.Process(target=visualization.viz_proc,
                                          args=(args.input, args.data, viz_queue, fps_calc.FPSCalc(2), nusc, demo_mng,
                                                'BEV demo - ground truth', 'GROUND TRUTH'))
    viz_process.start()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    feeder = threading.Thread(target=ground_truth.ground_truth_proc,
                              args=(viz_queue, scenes, boxes, demo_mng, args.run_slow, sweeps), daemon=True)
    feeder.start()

    try:
        while not demo_mng.get_terminate():
            time.sleep(1)
    except KeyboardInterrupt:
        demo_mng.set_terminate()
    finally:
        feeder.join()
        viz_process.join(timeout=5)
        if viz_process.is_alive():
            viz_process.terminate()
        viz_queue.cancel_join_thread()
        viz_queue.close()


if __name__ == "__main__":
    args = parse_args()
    args.input, args.models, args.data = (startup_checks.as_dir(path) for path in (args.input, args.models, args.data))
    startup_checks.check_input(args.input)
    with open(f'{args.input}/nusc_tiny_dataset.json', 'r', encoding='utf-8') as file:
        nusc = json.load(file)
    startup_checks.check_data(args.data, nusc)
    if not args.ground_truth:
        startup_checks.check_models(args.models)

    backbone_hef_path = f'{args.models}petrv2_repvggB0_backbone_pp_800x320.hef'
    transformer_hef_path = f'{args.models}/petrv2_repvggB0_transformer_pp_800x320.hef'
    post_proc_onnx_path = f'{args.models}/petrv2_postprocess.onnx'
    matmul_path = f'{args.models}/matmul.npy'

    fps_calculator = fps_calc.FPSCalc(2)

    queues = []
    bb_tranformer_meta_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    transformer_pp_meta_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    bb_tranformer_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    transformer_pp_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    pp_3dnms_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    nms_send_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)

    queues.append(bb_tranformer_meta_queue)
    queues.append(transformer_pp_meta_queue)
    queues.append(bb_tranformer_queue)
    queues.append(transformer_pp_queue)
    queues.append(pp_3dnms_queue)
    queues.append(nms_send_queue)

    with open(f'{args.input}tokens.json', 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    # Data prepared with --sweeps (~12 Hz) is played at the recorded pace
    sweeps = scenes.get('sweeps', False)
    scenes = scenes['scenes']
    timestamps = {token: timestamp for scene in scenes for token, timestamp in zip(scene['tokens'], scene.get('timestamps', []))}
    # The first sample of each scene (after its 'first_' duplicate), where the tracker restarts
    scene_starts = {scene['tokens'][1] for scene in scenes if len(scene['tokens']) > 1}

    tokens = []
    for scene in scenes:
        tokens.append(scene['tokens'])

    if args.ground_truth:
        run_ground_truth(args, nusc, scenes, sweeps)
        sys.exit(0)

    manager = multiprocessing.Manager()
    demo_mng = demo_manager.DemoManager(manager)
    devices = Device.scan()
    params = async_api.create_vdevice_params()
    threads = []
    processes = []
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    errors = []  # names of the pipeline threads or processes that failed
    with VDevice(params) as target:
        if args.raw_data:
            threads.append(threading.Thread(target=startup_checks.stop_demo_on_error(core.backbone_raw_data, demo_mng, errors), args=(target, args.input, backbone_hef_path, bb_tranformer_queue,
                                                    bb_tranformer_meta_queue, demo_mng, scenes, args.run_slow, sweeps)))
        else:
            threads.append(threading.Thread(target=startup_checks.stop_demo_on_error(core.backbone_from_jpg, demo_mng, errors), args=(target, args.data, backbone_hef_path, bb_tranformer_queue,
                                        bb_tranformer_meta_queue, demo_mng, scenes, args.run_slow, nusc, sweeps)))

        threads.append(threading.Thread(target=startup_checks.stop_demo_on_error(core.transformer, demo_mng, errors), args=(target, transformer_hef_path, matmul_path, bb_tranformer_queue, bb_tranformer_meta_queue, transformer_pp_queue, transformer_pp_meta_queue, demo_mng, timestamps, args.prev_frame_gap)))

        processes.append(multiprocessing.Process(target=pre_post_process.post_proc,
                                                args=(transformer_pp_queue, transformer_pp_meta_queue,
                                                pp_3dnms_queue, post_proc_onnx_path, demo_mng)))
        processes.append(multiprocessing.Process(target=pre_post_process.d3nms_proc,
                                                args=(pp_3dnms_queue, nms_send_queue, nusc, demo_mng, args.score_threshold, not args.no_tracker, timestamps, scene_starts)))
        processes.append(multiprocessing.Process(target=visualization.viz_proc,
                                            args=(args.input, args.data, nms_send_queue,
                                            fps_calculator, nusc, demo_mng)))

        try:
            # Ctrl-C is sent to every process in the group. Only the main process should handle it,
            # the child processes stop through demo_mng, otherwise they die mid-operation and leave the demo hanging.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            for process in processes:
                process.start()
            signal.signal(signal.SIGINT, signal.default_int_handler)

            for thread in threads:
                thread.daemon = True
                thread.start()

            while not demo_mng.get_terminate():
                time.sleep(1)
                # A pipeline process that crashed would leave the others waiting for its output forever
                failed = [process.name for process in processes if process.exitcode not in (None, 0)]
                if failed:
                    errors.extend(failed)
                    demo_mng.set_terminate()


        except KeyboardInterrupt:
            demo_mng.set_terminate()

        finally:
            for thread in threads:
                thread.join()

            # Release the inference models from the main thread, before the VDevice
            async_api.release_all()

            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()

            for queue_to_close in queues:
                # Don't block on exit flushing data that no one will read
                queue_to_close.cancel_join_thread()
                queue_to_close.close()

            target.release()

    if errors:
        sys.exit(f'Error: the demo stopped because {", ".join(errors)} failed, see the error above.')
