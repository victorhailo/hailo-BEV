# pylint: disable=C0301
# pylint: disable=E1101
# pylint: disable=C0411
# pylint: disable=R0914
# pylint: disable=R0913
# pylint: disable=C0200

from pyquaternion import Quaternion
import numpy as np
import cv2
import json
import math
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple


# Constants
COLOR_RED = (0, 0, 255)
COLOR_LIGHT_BLUE = (255, 160, 40)
COLOR_GRAY = (155, 155, 155)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 158, 255)
COLOR_DARK_BLUE = (60, 20, 220)
COLOR_ORANGE = (0, 80, 255)

LINE_WIDTH = 2
BOX_LINE_WIDTH = 2  # at the displayed camera image size (IMAGE_WIDTH)
COLOR_OUTLINE = (20, 20, 20)
FRAME_WIDTH = 60
FRAME_HEIGHT = 60
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 338
TOP_IMAGE_SIZE = (640, 640, 3)
NUM_OF_IMAGES = 6


def render_ego_centric_map(nusc,
                           sample_data_token: str,
                           axes_limit: float = 40) -> None:
    """
    Render map centered around the associated ego pose.
    :param sample_data_token: Sample_data token of the LIDAR_TOP record (keyframe or sweep) giving the ego pose.
    :param axes_limit: Axes limit measured in meters.
    :param ax: Axes onto which to render.
    """

    def crop_image(image: np.array,
                    x_px: int,
                    y_px: int,
                    axes_limit_px: int) -> np.array:
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px)
        y_max = int(y_px + axes_limit_px)

        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image

    # Get data.
    sd_record = nusc.get('sample_data', sample_data_token)
    sample = nusc.get('sample', sd_record['sample_token'])
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_ = nusc.get('map', log['map_token'])
    map_mask = map_['mask']
    pose = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Retrieve and crop mask.
    pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
    scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
    mask_raster = map_mask.mask()
    cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

    # Rotate image.
    ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    rotated_image = rotate_image(cropped, yaw_deg, cropped.shape[0])

    # Crop image.
    ego_centric_map = crop_image(rotated_image,
                                    int(rotated_image.shape[1] / 2),
                                    int(rotated_image.shape[0] / 2),
                                    scaled_limit_px)
    # Change colors.
    ego_centric_map[ego_centric_map == map_mask.background] = 254
    ego_centric_map[ego_centric_map == map_mask.foreground] = 125

    # Resize image.
    resized_image = cv2.resize(ego_centric_map, (640, 640))
    result_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
    return result_image

# Corner pattern of nuScenes Box.corners(): the first four corners face forward, the last four backward
CORNER_SIGNS = np.array([[1,  1,  1,  1, -1, -1, -1, -1],
                         [1, -1, -1,  1,  1, -1, -1,  1],
                         [1,  1, -1, -1,  1,  1, -1, -1]], dtype=float)

def quaternions_to_matrices(quaternions) -> np.ndarray:
    """
    Converts (N, 4) quaternions in (w, x, y, z) order to (N, 3, 3) rotation matrices.
    """
    q = np.asarray(quaternions, dtype=float).reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)], axis=1)

def boxes_corners(annos) -> np.ndarray:
    """
    Computes the corners of all annotated boxes at once, in the same order as nuScenes Box.corners().
    Building a nuScenes Box per annotation and per camera is much slower.

    Returns:
    - np.ndarray: (N, 3, 8) corners in global coordinates.
    """
    if not annos:
        return np.zeros((0, 3, 8))
    centers = np.array([anno['translation'] for anno in annos], dtype=float)
    w, l, h = np.array([anno['size'] for anno in annos], dtype=float).T
    half_sizes = np.stack([l, w, h], axis=1)[:, :, None] / 2
    rotations = quaternions_to_matrices([anno['rotation'] for anno in annos])
    return rotations @ (half_sizes * CORNER_SIGNS) + centers[:, :, None]

def draw_box(img, corners, color, linewidth, outline=None) -> None:
    """
    Draws a box with the same lines as nuScenes Box.render_cv2, with a single color.

    Parameters:
    - img (np.ndarray): Image to draw on, modified in-place.
    - corners (np.ndarray): (2, 8) corner positions in pixels.
    - color (tuple): Line color.
    - linewidth (int): Line width.
    - outline (tuple, optional): If given, the lines are first drawn wider in this color,
      so that they stay visible on both light and dark backgrounds.
    """
    points = [(int(x), int(y)) for x, y in corners.T]

    # The sides, the front (first 4 corners) and rear (last 4 corners) rectangles
    segments = [(points[i], points[i + 4]) for i in range(4)]
    for rect in (points[:4], points[4:]):
        segments += list(zip([rect[-1]] + rect[:-1], rect))

    # Line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    segments.append(((int(center_bottom[0]), int(center_bottom[1])),
                     (int(center_bottom_forward[0]), int(center_bottom_forward[1]))))

    if outline is not None:
        for start, end in segments:
            cv2.line(img, start, end, outline, linewidth + 2, cv2.LINE_AA)
    for start, end in segments:
        cv2.line(img, start, end, color, linewidth, cv2.LINE_AA)

def get_color(category_name: str) -> Tuple[int, int, int]:
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories,
    as well as the nuScenes detection categories.
    """
    if "human" in category_name:
        c = COLOR_LIGHT_BLUE
    elif "vehicle.car" in category_name or "vehicle.construction" in category_name:
        c = COLOR_YELLOW
    elif "vehicle.bicycle" in category_name or "vehicle.motorcycle" in category_name:
        c = COLOR_DARK_BLUE
    else:
        c = COLOR_ORANGE
    return c

def rgb_2_bgr(color) -> Tuple[int, int, int]:
    """
    Converts rgb color to bgr color
    """
    r = color[0]
    g = color[1]
    b = color[2]
    return (b, g, r)

def rotate_image(bev_image, degrees, size) -> np.ndarray:
    """
    Rotates an image by degrees
    """
    rotation_matrix = cv2.getRotationMatrix2D((int(size/2), int(size/2)), degrees, 1)
    rotated_image = cv2.warpAffine(bev_image, rotation_matrix, (size, size))
    return rotated_image

def show_bev(bev_image, annos, sample_token, nusc) -> np.ndarray:
    """
    Renders annotated bounding boxes on a top-view radar (bird's-eye view) image using NuScenes dataset annotations.

    Parameters:
    - bev_image (np.ndarray): Background bird's-eye view image (e.g., from radar or lidar) in numpy array format.
    - annos (list of dict): List of annotation dictionaries for the objects to be rendered.
      Each annotation contains:
        - 'translation' (list): Position of the object in 3D space.
        - 'size' (list): Dimensions of the bounding box for the object.
        - 'rotation' (list): Quaternion representing the object's rotation.
        - 'category_name' (str): Name of the object category.
        - 'token' (str): Unique identifier for the annotation.
    - sample_token (str): Unique identifier for the specific sample in the NuScenes dataset.
    - nusc (NuScenes): NuScenes dataset instance providing access to dataset objects and metadata.

    Returns:
    - np.ndarray: The annotated bird's-eye view image, rotated 90 degrees clockwise for correct orientation.
    """
    pose_record = nusc[sample_token]['LIDAR_TOP'][0]

    # Global -> ego coordinates, rotating by the ego yaw only so the view stays level
    yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
    yaw_inverse = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse.rotation_matrix
    corners = yaw_inverse @ (boxes_corners(annos) - np.array(pose_record['translation'])[:, None])

    # Top view: keep x and y, flip y, then shift and scale to pixels
    corners = corners[:, :2, :]
    corners[:, 1, :] *= -1
    corners = (corners + 40) * 8

    for anno, box_corners in zip(annos, corners):
        if "movable_object" not in anno['category_name']:
            draw_box(bev_image, box_corners, get_color(anno['category_name'])[::-1], LINE_WIDTH)
    rotated_image = rotate_image(bev_image, 90, bev_image.shape[0])
    return rotated_image

def show_image_with_boxes(sample_token, cv2_image, cam, annos, nusc) -> None:
    """
    Renders annotated bounding boxes on a camera image using NuScenes dataset annotations.

    Parameters:
    - sample_token (str): Unique identifier for the specific sample in the NuScenes dataset.
    - cv2_image (np.ndarray): Camera image in OpenCV format (BGR color order) on which the
      bounding boxes will be drawn.
    - cam (str): Camera channel name (e.g., 'CAM_FRONT') indicating which camera's data to use.
    - annos (list of dict): List of annotation dictionaries for the objects to be rendered.
      Each annotation contains:
        - 'translation' (list): Position of the object in 3D space.
        - 'size' (list): Dimensions of the bounding box for the object.
        - 'rotation' (list): Quaternion representing the object's rotation.
        - 'category_name' (str): Name of the object category.
        - 'token' (str): Unique identifier for the annotation.
    - nusc (NuScenes): NuScenes dataset instance providing access to dataset objects and metadata.

    Returns:
    - None: The function modifies the `cv2_image` in-place by drawing the bounding boxes,
      without returning any value.
    # """

    pose_record = nusc[sample_token][cam][0]
    cs_record = nusc[sample_token][cam][1]

    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    # The intrinsics are for the original 1600x900 image; rescale them to the size actually decoded
    scale = cv2_image.shape[1] / 1600
    cam_intrinsic[:2] *= scale
    width, height = cv2_image.shape[1], cv2_image.shape[0]
    line_width = max(1, round(BOX_LINE_WIDTH * cv2_image.shape[1] / IMAGE_WIDTH))

    # Global -> ego vehicle -> camera coordinates, for all boxes at once
    corners = boxes_corners(annos)
    corners = Quaternion(pose_record['rotation']).rotation_matrix.T @ (corners - np.array(pose_record['translation'])[:, None])
    corners = Quaternion(cs_record['rotation']).rotation_matrix.T @ (corners - np.array(cs_record['translation'])[:, None])

    # Project to pixels
    depth = corners[:, 2, :]
    projected = cam_intrinsic @ corners
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = projected[:, :2, :] / projected[:, 2:3, :]

    # Same rule as nuScenes box_in_image with BoxVisibility.ANY: at least one corner inside the image
    # and more than 1m in front of the camera, and all corners at least 0.1m in front of the camera
    visible = ((pixels[:, 0, :] > 0) & (pixels[:, 0, :] < width) &
               (pixels[:, 1, :] > 0) & (pixels[:, 1, :] < height) & (depth > 1))
    in_image = visible.any(axis=1) & (depth > 0.1).all(axis=1)

    for anno, box_pixels, show in zip(annos, pixels, in_image):
        if show and "movable_object" not in anno['category_name']:
            draw_box(cv2_image, box_pixels, get_color(anno['category_name'])[::-1], line_width, COLOR_OUTLINE)

def combine_images(images, bev_image, fps_calc) -> np.ndarray:
    """
    Combines a set of 6 camera images and a bird's-eye view (BEV) image into a single composite image,
    with camera labels and FPS information.

    Parameters:
    - images (list of np.ndarray): List of 6 camera images in OpenCV format (BGR color order).
      Each image is expected to have the same dimensions.
    - bev_image (np.ndarray): Bird's-eye view (BEV) image as a numpy array in BGR format.
    - fps_calc (object): Object with a `get_fps()` method that returns the current FPS value.

    Returns:
    - np.ndarray: A single composite image containing:
        - 6 camera images arranged in a 2x3 grid, labeled by camera positions.
        - The BEV image, positioned alongside the camera images.
        - FPS information displayed at a specified position on the composite image.

    Raises:
    - ValueError: If the number of images provided is not exactly 6.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (0, 0, 0)
    # Ensure that there are exactly 6 images
    # assert len(images) != NUM_OF_IMAGES, f'Please provide exactly {NUM_OF_IMAGES} images.'

    image_dimensions = [img.size for img in images]
    # assert len(set(image_dimensions)) > 1, "Error: All images must have the same dimensions."

    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        resized_images.append(resized_image)

    # Assuming all images have the same dimensions, get the width and height of one image
    img_width, img_height = resized_images[0].shape[1], resized_images[0].shape[0]
    total_width = 3 * (img_width + 2 * FRAME_WIDTH) + bev_image.shape[0]
    total_height = 2 * (img_height + 2 * FRAME_HEIGHT)

    combined_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    # Convert the resized images to NumPy arrays and paste onto combined_image
    for i in range(2):
        for j in range(3):
            index = i * 3 + j
            x_offset = j * (img_width + FRAME_WIDTH) + FRAME_WIDTH
            y_offset = i * (img_height + FRAME_HEIGHT) + FRAME_HEIGHT
            img_np = resized_images[index]
            combined_image[y_offset:y_offset+img_height, x_offset:x_offset+img_width, :] = img_np

            # Draw text directly on combined_image
            text = f'CAM {["FRONT LEFT", "FRONT", "FRONT RIGHT", "BACK LEFT", "BACK", "BACK RIGHT"][index]}'
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = x_offset + (img_width - text_size[0]) // 2
            text_y = y_offset - 10  # Place text 10 pixels above the image
            cv2.putText(combined_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    bev_height, bev_width, _ = bev_image.shape
    x_offset = total_width - bev_width
    y_offset = int((total_height - bev_height)/2)
    combined_image[y_offset:y_offset+bev_height, x_offset:x_offset+bev_width, :] = bev_image
    combined_image[y_offset, x_offset:, :] = [255, 255, 255]

    return combined_image

def read_cam_image(path) -> np.ndarray:
    # Decode at half resolution (800x450), which is much faster, then shrink to the displayed size,
    # so that the boxes are drawn at the final resolution instead of being thinned by a later resize
    image = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if image is None:
        raise FileNotFoundError(f'Cannot read image {path}')
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def render_frame(input_path, data_path, annos, token, nusc, fps_calc, read_pool) -> np.ndarray:
    """
    Renders the 6 camera images and the BEV image with the detected boxes into one BGR frame.
    """
    cams = ['CAM_FRONT_LEFT','CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    cam_images = list(read_pool.map(read_cam_image, [data_path + nusc[token][cam][2]['filename'] for cam in cams]))
    bev_image = cv2.imread(input_path + nusc[token]['LIDAR_TOP'][2]['filename'])
    if bev_image is None:
        raise FileNotFoundError(f"Cannot read map image {input_path + nusc[token]['LIDAR_TOP'][2]['filename']}")

    cv2.rectangle(bev_image, (int(TOP_IMAGE_SIZE[0]/2) - 20,
                                int(TOP_IMAGE_SIZE[0]/2) - 10),
                                (int(TOP_IMAGE_SIZE[0]/2) + 20,
                                int(TOP_IMAGE_SIZE[0]/2) + 10),
                                COLOR_GREEN, thickness=LINE_WIDTH)

    top_image = show_bev(bev_image,annos,token, nusc)
    for cam_image, cam in zip(cam_images, cams):
        show_image_with_boxes(token, cam_image, cam, annos, nusc)

    combine_img = combine_images(cam_images, top_image, fps_calc)
    return cv2.cvtColor(np.array(combine_img), cv2.COLOR_RGB2BGR)

# Playback controls. Arrow key codes returned by cv2.waitKeyEx differ between GUI backends:
# X11 keysyms (Qt and GTK on Linux), Windows virtual key codes, and Qt key codes
KEYS_PREVIOUS = {65361, 2424832, 0x1000012}
KEYS_NEXT = {65363, 2555904, 0x1000014}
KEYS_PLAY_PAUSE = {ord(' ')}
KEYS_QUIT = {ord('q'), ord('Q'), 27}
HISTORY_SIZE = 300

# Clickable buttons drawn in the white strip under the camera images: (action, x, y, width, height)
BUTTON_SIZE = (90, 60)
BUTTONS_Y = 2 * (IMAGE_HEIGHT + 2 * FRAME_HEIGHT) - FRAME_HEIGHT - 45
BUTTONS = [('previous', 870, BUTTONS_Y), ('play_pause', 975, BUTTONS_Y), ('next', 1080, BUTTONS_Y)]

def draw_controls(frame, paused, status, label='') -> None:
    """
    Draws the playback buttons and a status line in the bottom margin of the frame (modified in-place).
    """
    width, height = BUTTON_SIZE
    for action, x, y in BUTTONS:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (225, 225, 225), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (90, 90, 90), 2)
        cx, cy = x + width // 2, y + height // 2
        if action == 'previous':
            shape = [[(cx + 15, cy - 18), (cx - 15, cy), (cx + 15, cy + 18)], [(cx - 20, cy - 18), (cx - 15, cy - 18), (cx - 15, cy + 18), (cx - 20, cy + 18)]]
        elif action == 'next':
            shape = [[(cx - 15, cy - 18), (cx + 15, cy), (cx - 15, cy + 18)], [(cx + 15, cy - 18), (cx + 20, cy - 18), (cx + 20, cy + 18), (cx + 15, cy + 18)]]
        elif paused:
            shape = [[(cx - 12, cy - 18), (cx + 16, cy), (cx - 12, cy + 18)]]
        else:
            shape = [[(cx - 13, cy - 18), (cx - 4, cy - 18), (cx - 4, cy + 18), (cx - 13, cy + 18)],
                     [(cx + 4, cy - 18), (cx + 13, cy - 18), (cx + 13, cy + 18), (cx + 4, cy + 18)]]
        cv2.fillPoly(frame, [np.array(points) for points in shape], (40, 40, 40))

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_y = BUTTONS_Y + height // 2 + 10
    cv2.putText(frame, label, (FRAME_WIDTH, BUTTONS_Y + 18), font, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, status, (FRAME_WIDTH, BUTTONS_Y + 55), font, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, 'Space: play/pause   Left/Right: step   Q: quit', (1220, text_y), font, 0.8, (90, 90, 90), 2)

def viz_proc(input_path, data_path, in_queue, fps_calc, nusc, demo=None,
             title='BEV demo - Hailo-8', label='HAILO-8 DETECTIONS') -> None:
    """
    Visualization process that handles displaying images, annotations, and 3D bounding boxes in a window.

    This function fetches annotations for a given token from the `in_queue`, and renders the camera images
    with bounding boxes and the bird's-eye view in a window. It also handles FPS calculation and updating.

    The last frames are kept, so that playback can be paused and stepped through with the keyboard
    (Space, Left/Right arrows) or with the buttons under the images. While paused, the pipeline
    upstream waits, since its queues are full.

    Args:
        in_queue (Queue): A queue that provides camera images and annotation data (in the form of a tuple `(annos, token)`).
        fps_calc (fps_calc.FPSCalc): An instance of the `FPSCalc` class used to calculate and update FPS.
        nusc (NuScenes): An instance of the `NuScenes` class to fetch data from the nuScenes dataset (camera images, annotations, etc.).
        demo (object, optional): A demo object that handles termination signals. If not given, the function runs
        until the process is terminated or the window is closed.
        title (str): Window title, to tell instances apart when several run side by side.
        label (str): Source of the boxes, shown under the images with the scene and frame number.

    Returns:
        None: The function runs until the termination signal is received from the `demo` object,
        or until the user quits.
    """
    # Scene and frame number of each sample, to line up instances showing the same data
    with open(f'{input_path}/tokens.json', 'r', encoding='utf-8') as file:
        scenes = json.load(file)['scenes']
    positions = {}
    for scene_index, scene in enumerate(scenes):
        frames = [token for token in scene['tokens'] if not token.startswith('first')]
        for frame_index, token in enumerate(frames):
            positions[token] = f'scene {scene_index + 1}/{len(scenes)}   frame {frame_index + 1}/{len(frames)}'

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
    # The frame is 2800x916; open the window at half size instead of Qt's default 400x300
    cv2.resizeWindow(title, 1400, 458)

    clicks = []
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, bx, by in BUTTONS:
                if bx <= x <= bx + BUTTON_SIZE[0] and by <= y <= by + BUTTON_SIZE[1]:
                    clicks.append(action)
    cv2.setMouseCallback(title, on_mouse)

    # cv2.imread releases the GIL, so the 6 camera images can be decoded in parallel
    read_pool = ThreadPoolExecutor(max_workers=6)
    history = deque(maxlen=HISTORY_SIZE)  # (sequence number, annos, token) of the last frames received
    received = 0
    position = -1  # index in history of the frame on screen
    paused = False
    step_pending = False
    shown = None  # (sequence number, paused) of the frame on screen, to redraw only when it changes
    frame = None

    while demo is None or not demo.get_terminate():
        key = cv2.waitKeyEx(30 if paused else 1)
        try:
            window_closed = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            window_closed = True
        if key in KEYS_QUIT or window_closed:
            if demo is not None:
                demo.set_terminate()
            break

        actions = clicks[:]
        clicks.clear()
        if key in KEYS_PREVIOUS:
            actions.append('previous')
        elif key in KEYS_NEXT:
            actions.append('next')
        elif key in KEYS_PLAY_PAUSE:
            actions.append('play_pause')
        for action in actions:
            if action == 'play_pause':
                paused = not paused
                if not paused:
                    fps_calc.restart()
            elif action == 'previous':
                paused = True
                if position > 0:
                    position -= 1
            elif action == 'next':
                paused = True
                step_pending = True

        new_frame = False
        if not paused or step_pending:
            if position < len(history) - 1:
                # Replay frames the user stepped back over
                position += 1
                step_pending = False
            else:
                try:
                    annos, token = in_queue.get(block=True, timeout=0.1)
                except queue.Empty:
                    continue
                history.append((received, annos, token))
                received += 1
                position = len(history) - 1
                step_pending = False
                new_frame = True

        if position < 0:
            continue
        sequence, annos, token = history[position]
        if shown == (sequence, paused):
            continue
        if shown is None or shown[0] != sequence:
            frame = render_frame(input_path, data_path, annos, token, nusc, fps_calc, read_pool)
        frames_back = len(history) - 1 - position
        if paused:
            status = 'PAUSED' + (f'   ({frames_back} back)' if frames_back else '')
        elif frames_back:
            status = f'REPLAY   {frames_back} frame{"s" if frames_back > 1 else ""} behind live'
        else:
            status = f'LIVE   {float(fps_calc.get_fps()):.1f} FPS'
        shown_frame = frame.copy()
        draw_controls(shown_frame, paused, status, f'{label}   {positions.get(token, "")}')
        cv2.imshow(title, shown_frame)
        shown = (sequence, paused)
        if new_frame:
            fps_calc.update_fps()

    read_pool.shutdown()
    cv2.destroyAllWindows()
