# pylint: disable=C0301
"""
Checks run before the demo starts, so that a wrong path fails right away with a clear message,
instead of a pipeline thread failing on its first file while the demo waits for frames forever.
"""
import os
import sys
import traceback

CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
MODEL_FILES = ['petrv2_repvggB0_backbone_pp_800x320.hef', 'petrv2_repvggB0_transformer_pp_800x320.hef',
               'petrv2_postprocess.onnx', 'matmul.npy']

def as_dir(path) -> str:
    """
    Returns the path with a trailing separator: the demo builds file paths by appending file names to it.
    """
    return os.path.join(path, '')

def check_input(input_path) -> None:
    """
    Exits with an error message if the input folder created by prepare_data.py is missing.
    """
    for name in ['nusc_tiny_dataset.json', 'tokens.json']:
        if not os.path.isfile(input_path + name):
            sys.exit(f'Error: {input_path + name} not found.\n'
                     f'Create the input folder with ./src/common/prepare_data.py --data <path to nuScenes dataset>, '
                     f'or pass its location with -i.')

def check_data(data_path, nusc) -> None:
    """
    Exits with an error message if the camera images of the prepared data are not in the nuScenes data folder.

    Args:
        data_path (str): Path to the nuScenes dataset.
        nusc (dict): The prepared nusc_tiny_dataset.json.
    """
    if not os.path.isdir(data_path):
        sys.exit(f'Error: nuScenes data folder {data_path} not found.\n'
                 f'Pass the nuScenes dataset location with -d, e.g. -d /path/to/v1.0-mini/')
    first_token = next(iter(nusc))
    for cam in CAMS:
        image_path = data_path + nusc[first_token][cam][2]['filename']
        if not os.path.isfile(image_path):
            sys.exit(f'Error: camera image {image_path} not found.\n'
                     f'The -d folder must be the nuScenes dataset the input folder was prepared from '
                     f'(the folder that contains samples/ and sweeps/).')

def check_models(models_path) -> None:
    """
    Exits with an error message if a model file downloaded by download_resources.sh is missing.
    """
    for name in MODEL_FILES:
        if not os.path.isfile(models_path + name):
            sys.exit(f'Error: model file {models_path + name} not found.\n'
                     f'Download the models with ./download_resources.sh, or pass their location with -m.')

def stop_demo_on_error(target, demo, errors):
    """
    Wraps a pipeline thread function: if it raises, the error is printed, recorded in `errors`,
    and the whole demo is stopped, instead of the other stages waiting for its output forever.
    """
    def run(*args, **kwargs):
        try:
            target(*args, **kwargs)
        except Exception:  # pylint: disable=W0703
            errors.append(target.__name__)
            traceback.print_exc()
            demo.set_terminate()
    return run
