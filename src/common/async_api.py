# pylint: disable=E0401
import queue
import threading
import time
from functools import partial
from loguru import logger
import numpy as np
from hailo_platform import (HEF, VDevice, FormatType, HailoSchedulingAlgorithm)
from hailo_platform.pyhailort.pyhailort import HailoRTTimeout


def create_vdevice_params():
    """
    Create parameters for a virtual device.
    """
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.multi_process_service = False
    return params

# Every HailoAsyncInference created, kept referenced until release_all()
_instances = []

class _JobCounter:
    """
    Counts the inference jobs whose callback has not finished yet.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.count = 0

    def add(self, value):
        with self._lock:
            self.count += value

def release_all(timeout=5.0) -> None:
    """
    Waits for pending jobs, then releases every HailoAsyncInference from the calling thread.
    Call it from the main thread, after the inference threads are joined and before releasing the VDevice.

    HailoRT's Python completion callback keeps a reference to the ConfiguredInferModel. If the last reference
    is dropped on HailoRT's callback thread, the model's destructor tries to join that thread from itself and
    the process aborts ("Resource deadlock avoided"). Keeping every model referenced here until its jobs are
    done makes sure it is destroyed on the calling thread instead.
    (ConfiguredInferModel.shutdown() is not called: it sometimes blocks forever at this point.)

    Args:
        timeout (float): Max time to wait for the pending jobs of each model, in seconds.
    """
    for inference in _instances:
        deadline = time.time() + timeout
        while inference.pending_jobs.count > 0 and time.time() < deadline:
            time.sleep(0.01)
        if inference.pending_jobs.count > 0:
            logger.warning(f'{inference.pending_jobs.count} inference jobs still pending after {timeout} s')
    # Our callback is done, give HailoRT's callback wrapper time to return and drop its reference
    time.sleep(0.2)
    _instances.clear()

class HailoAsyncInference:
    def __init__(self, target, hef_path, queue_out, demo,
                 input_names, output_names, batch_size=1, output_type='FLOAT32', input_type='FLOAT32'):
        """
        Initialize the HailoAsyncInference class with the provided HEF model file path.

        Args:
            hef_path (str): Path to the HEF model file.
            batch_size (int): Batch size for inference.
            output_type (str): Format type of the output stream.
        """

        self.hef = HEF(hef_path)
        self.demo = demo
        self.target = target
        self.input_names = input_names
        self.output_names = output_names
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)
        self.input_type = input_type
        self._set_input_output(output_type, input_type)
        self.input_vstream_info, self.output_vstream_info = self._get_vstream_info()
        self.configured_infer_model = self.infer_model.configure()
        self.queue = queue_out
        self.batch_size = batch_size
        self.pending_jobs = _JobCounter()
        _instances.append(self)


    def _set_input_output(self, output_type, input_type):
        """
        Set the input and output layer information for the HEF model.

        Args:
            output_type (str): Format type of the output stream.
        """
        for name in self.input_names:
            self.infer_model.input(name).set_format_type(getattr(FormatType, input_type))
        for name in self.output_names:
            self.infer_model.output(name).set_format_type(getattr(FormatType, output_type))

    @staticmethod
    def callback(completion_info, all_bindings, out_queue, output_names, demo, pending_jobs):
        """
        Callback function for handling inference results.

        It is static so that it holds no reference to the HailoAsyncInference object, see release_all().

        Args:
            completion_info: Information about the completion of the inference task.
            bindings: Bindings object containing input and output buffers.
        """
        try:
            HailoAsyncInference._handle_results(completion_info, all_bindings, out_queue, output_names, demo)
        finally:
            pending_jobs.add(-1)

    @staticmethod
    def _handle_results(completion_info, all_bindings, out_queue, output_names, demo):
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            outputs = {}
            for name in output_names:
                outputs[name] = []
                for binding in all_bindings:
                    outputs[name].append(binding.output(name).get_buffer())

            while not demo.get_terminate():
                try:
                    out_queue.put(outputs, block=True, timeout=0.5)
                    break
                except queue.Full:
                    pass


    def _get_vstream_info(self):
        """
        Get information about input and output stream layers.

        Returns:
            tuple: List of input stream layer information, List of output stream layer information.
        """
        return self.hef.get_input_vstream_infos(), self.hef.get_output_vstream_infos()

    def run(self, input_data):
        """
        Run asynchronous inference on the Hailo-8 device.

        Args:
            input_data (np.ndarray): Input data for inference.

        Returns:
            AsyncInferJob: The async inference job, or None if the demo was terminated while waiting.
        """

        all_bindings = []
        for batch in range(self.batch_size):
            bindings = self._create_bindings()
            for name in self.input_names:
                bindings.input(name).set_buffer(input_data[name][batch])

            all_bindings.append(bindings)

        # Wait for room for the whole batch: submitting more frames than the queue can hold
        # fails with HAILO_QUEUE_IS_FULL and shuts down the inference pipeline
        while True:
            try:
                self.configured_infer_model.wait_for_async_ready(timeout_ms=1000, frames_count=len(all_bindings))
                break
            except HailoRTTimeout:
                if self.demo.get_terminate():
                    return None

        self.pending_jobs.add(1)
        try:
            job = self.configured_infer_model.run_async(all_bindings,
                                partial(HailoAsyncInference.callback, all_bindings=all_bindings, out_queue=self.queue,
                                        output_names=self.output_names, demo=self.demo, pending_jobs=self.pending_jobs))
        except Exception:
            self.pending_jobs.add(-1)
            raise

        return job

    def _create_bindings(self):
        """
        Create bindings for input and output buffers.

        Returns:
            bindings: Bindings object with input and output buffers.
        """
        output_buffers = {name: np.empty(self.infer_model.output(name).shape, dtype=np.float32)
                          for name in self.infer_model.output_names}
        return self.configured_infer_model.create_bindings(output_buffers=output_buffers)
