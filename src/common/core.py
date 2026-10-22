import queue
import numpy as np
import async_api
import os
import pre_post_process
import pacing

# Default time between the current frame and the earlier frame given to the model with it, in seconds
DEFAULT_PREV_FRAME_GAP = 0.25

def backbone_raw_data(target, input_path, hef_path, queue_out, queue_meta_out, demo, scenes, run_slow, realtime=False) -> None:
    """
    Runs inference on the backbone model asynchronously, processing a series of images from the nuScenes dataset.

    This function handles the loading and preprocessing of images from the nuScenes dataset,
    performs asynchronous inference on the specified backbone model using the HailoAsyncInference API,
    and manages the flow of metadata and results through the provided queues. The function runs continuously
    in a loop, processing scenes and samples, and controlling the inference speed based on the `run_slow` flag.

    Args:
        target (str): The target device for inference (e.g., a Hailo device).
        data_path (str): The path to the dataset containing the image files.
        hef_path (str): The path to the Hailo Execution Flow (HEF) file.
        queue_out (Queue): The output queue for inference results.
        queue_meta_out (Queue): The output queue for metadata.
        demo (object): A demo object that handles termination signals and visualization.
        run_slow (bool): If `True`, the inference runs at 5 FPS, slowing the processing speed.
        realtime (bool): If `True`, the frames are not sent faster than they were recorded (for the ~12 Hz sweeps).
        nusc (object): The nuScenes dataset object that provides access to samples and scene data.

    Returns:
        None: This function runs indefinitely until the termination signal is received from the `demo` object.
    """
    assert os.path.exists(hef_path), f"File not found: {hef_path}"

    hailo_inference = async_api.HailoAsyncInference(target, hef_path, queue_out, demo,
                                          ['petrv2_repvggB0_backbone_pp_800x320/input_layer1'],
                                          ['petrv2_repvggB0_backbone_pp_800x320/conv28'], 6,  output_type='FLOAT32', input_type='UINT8')
    tensor_data = []
    for scene in scenes:
        assert 'input' in scene, "Please run ./prepare_data.py script with --raw-data flag"
        input_file_name = scene['input']
        # Memory-mapped: the frames are read from disk when used, instead of loading every scene into RAM
        tensor_data.append(np.load(f'{input_path}/{input_file_name}', mmap_mode='r'))

    pacer = pacing.FramePacer(5 if run_slow else None)
    while True:
        for scene, tensor_datas in zip(scenes, tensor_data):
            for i, token in enumerate(scene['tokens']):
                pacer.wait(scene['timestamps'][i] if realtime else None)

                job = hailo_inference.run({'petrv2_repvggB0_backbone_pp_800x320/input_layer1': np.array(tensor_datas[i])})
                while not demo.get_terminate():
                    try:
                        queue_meta_out.put(token, block=True, timeout=0.5)
                        break
                    except queue.Full:
                        pass

                if demo.get_terminate():
                    if job is not None:
                        job.wait(100000)
                    return

def backbone_from_jpg(target, data_path, hef_path, queue_out, queue_meta_out, demo, scenes, run_slow, nusc, realtime=False) -> None:
    """
    Runs inference on the backbone model asynchronously, processing a series of images from the nuScenes dataset.

    This function handles the loading and preprocessing of images from the nuScenes dataset,
    performs asynchronous inference on the specified backbone model using the HailoAsyncInference API,
    and manages the flow of metadata and results through the provided queues. The function runs continuously
    in a loop, processing scenes and samples, and controlling the inference speed based on the `run_slow` flag.

    Args:
        target (str): The target device for inference (e.g., a Hailo device).
        data_path (str): The path to the dataset containing the image files.
        hef_path (str): The path to the Hailo Execution Flow (HEF) file.
        queue_out (Queue): The output queue for inference results.
        queue_meta_out (Queue): The output queue for metadata.
        demo (object): A demo object that handles termination signals and visualization.
        run_slow (bool): If `True`, the inference runs at 5 FPS, slowing the processing speed.
        realtime (bool): If `True`, the frames are not sent faster than they were recorded (for the ~12 Hz sweeps).
        nusc (object): The nuScenes dataset object that provides access to samples and scene data.

    Returns:
        None: This function runs indefinitely until the termination signal is received from the `demo` object.
    """
    assert os.path.exists(hef_path), f"File not found: {hef_path}"

    hailo_inference = async_api.HailoAsyncInference(target, hef_path, queue_out, demo,
                                          ['petrv2_repvggB0_backbone_pp_800x320/input_layer1'],
                                          ['petrv2_repvggB0_backbone_pp_800x320/conv28'], 6,  output_type='FLOAT32', input_type='UINT8')
    pacer = pacing.FramePacer(5 if run_slow else None)
    while True:
        for scene in scenes:
            for i, token in enumerate(scene['tokens']):

                file_paths = []
                for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                    file_paths.append(nusc[token][cam][2]['filename'])
                pacer.wait(scene['timestamps'][i] if realtime else None)

                job = hailo_inference.run({'petrv2_repvggB0_backbone_pp_800x320/input_layer1': pre_post_process.preprocess(data_path, file_paths)})
                while not demo.get_terminate():
                    try:
                        queue_meta_out.put(token, block=True, timeout=0.5)
                        break
                    except queue.Full:
                        pass

                if demo.get_terminate():
                    if job is not None:
                        job.wait(100000)
                    return

def transformer(target, hef_path, matmul_path, queue_in,
                queue_meta_in, queue_out, queue_meta_out, demo, timestamps=None, prev_frame_gap=None) -> None:
    """
    Performs the transformer model's asynchronous inference step in the pipeline.

    This function handles the processing of input data for the transformer model.
    It takes the output from the backbone model, concatenates it with previous data (if available),
    performs inference on the transformer model using HailoAsyncInference,
    and sends the results to the next queue. It also manages metadata
    and ensures that inference proceeds while respecting termination signals.

    Args:
        target (str): The target device for inference (e.g., a Hailo device).
        hef_path (str): The path to the Hailo Execution Flow (HEF) file for the transformer model.
        matmul_path (str): Path to the matrix multiplication file required by the transformer model.
        queue_in (Queue): Input queue containing the processed data from the backbone.
        queue_meta_in (Queue): Input queue containing the metadata for the samples being processed.
        queue_out (Queue): Output queue for sending the transformer inference results.
        queue_meta_out (Queue): Output queue for sending the metadata after processing.
        demo (object): A demo object that handles termination signals and other demo-specific functionality.
        timestamps (dict, optional): Sample token -> capture time in seconds.
        prev_frame_gap (float, optional): Time between the current frame and the earlier frame given to the model
            with it, in seconds. The frame of the scene closest to that time is used (at the start of a scene, its
            first frame). Without it or without timestamps, the frame just before is used.

    Returns:
        None: This function runs indefinitely until the termination signal is received.
    """
    # Making sure that the files exist 
    assert os.path.exists(hef_path), f"File not found: {hef_path}"
    assert os.path.exists(matmul_path), f"File not found: {matmul_path}"
    
    # Initialize the HailoAsyncInference object for asynchronous inference
    hailo_inference = async_api.HailoAsyncInference(target, hef_path, queue_out, demo,
                                          ['petrv2_repvggB0_transformer_pp_800x320/input_layer1',
                                           'petrv2_repvggB0_transformer_pp_800x320/input_layer2'],
                                          ['petrv2_repvggB0_transformer_pp_800x320/concat1',
                                           'petrv2_repvggB0_transformer_pp_800x320/conv41'], 1)
    # Load the matrix multiplication data required for the transformer
    matmul = np.load(matmul_path)
    assert matmul.shape == (1, 12, 250, 256), "Expected shape of matmul is (1, 12, 250, 256), but got {}".format(matmul.shape)

    job = None
    history = []  # (capture time, backbone output) of the recent frames of the current scene
    while True:
        while not demo.get_terminate():
            try:
                meta_data = queue_meta_in.get(block=True, timeout=0.5)
                break
            except queue.Empty:
                pass
        while not demo.get_terminate():
            try:
                in_data = queue_in.get(block=True, timeout=0.5)
                break
            except queue.Empty:
                pass

        # Don't run inference on stale or missing data after termination
        if demo.get_terminate():
            if job is not None:
                job.wait(100000)
            break

        mid1_input = in_data['petrv2_repvggB0_backbone_pp_800x320/conv28']
        assert len(mid1_input) == 6 and mid1_input[0].shape == (10, 25, 1280),"Expected shape of mid1_input is (6, 12, 25, 1280), but got {}X{}".format(len(mid1_input),mid1_input[0].shape)

        timestamp = timestamps.get(meta_data.removeprefix('first_')) if timestamps else None
        if meta_data.startswith('first'):
            # Start of a scene: this frame only serves as the previous frame of the next ones
            history = [(timestamp, mid1_input)]
        else:
            if prev_frame_gap is None or timestamp is None:
                prev_mid1 = history[-1][1]
            else:
                prev_mid1 = min(history, key=lambda entry: abs(entry[0] - (timestamp - prev_frame_gap)))[1]
            mid2_input = np.concatenate((mid1_input,prev_mid1),axis=0)
            result = {}
            mid2_input = mid2_input.transpose(3, 0, 1, 2).reshape(1, 1280, 12, -1)
            mid2_input = np.transpose(mid2_input, (0, 2, 3, 1))

            result['petrv2_repvggB0_transformer_pp_800x320/input_layer1'] = mid2_input
            result['petrv2_repvggB0_transformer_pp_800x320/input_layer2'] = matmul

            job = hailo_inference.run(result)
            history.append((timestamp, mid1_input))
            # Keep only the frames that can still be used as previous frame
            if prev_frame_gap is None or timestamp is None:
                history = history[-1:]
            else:
                history = [entry for entry in history if entry[0] >= timestamp - prev_frame_gap - 0.5] or history[-1:]
            while not demo.get_terminate():
                try:
                    queue_meta_out.put(meta_data, block=True, timeout=0.5)
                    break
                except queue.Full:
                    pass

        if demo.get_terminate():
            if job is not None:
                job.wait(100000)
            break
