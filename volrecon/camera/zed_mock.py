"""Mock pyzed.sl for CI without physical ZED hardware."""

from __future__ import annotations

from enum import Enum

import numpy as np


class RESOLUTION(Enum):
    HD2K = 0
    HD1080 = 1
    HD720 = 2
    VGA = 3


class DEPTH_MODE(Enum):
    NONE = 0


class VIEW(Enum):
    LEFT = 0
    RIGHT = 1


class MEM(Enum):
    CPU = 0


class REFERENCE_FRAME(Enum):
    WORLD = 0


class UNIT(Enum):
    METER = 0


class POSITIONAL_TRACKING_MODE(Enum):
    GEN_1 = 0


class ERROR_CODE(Enum):
    SUCCESS = 0
    END_OF_SVOFILE = 1


RESOLUTION_MAP = {
    "HD2K": RESOLUTION.HD2K,
    "HD1080": RESOLUTION.HD1080,
    "HD720": RESOLUTION.HD720,
    "VGA": RESOLUTION.VGA,
}


class InitParameters:
    def __init__(self) -> None:
        self.camera_resolution = RESOLUTION.HD720
        self.camera_fps = 15
        self.coordinate_units = UNIT.METER
        self.depth_mode = DEPTH_MODE.NONE
        self.input = None
        self.svo_real_time_mode = False
        self.set_from_serial_number = None


class RuntimeParameters:
    pass


class Mat:
    def __init__(self) -> None:
        self._data: np.ndarray | None = None

    def get_data(self) -> np.ndarray:
        return self._data

    def set_data(self, data: np.ndarray) -> None:
        self._data = data


class CameraInformation:
    def __init__(self) -> None:
        self.serial_number = 12345678
        self.camera_model = "ZED2i"


class CalibrationParameters:
    class CamParams:
        def __init__(self, w: int, h: int) -> None:
            self.fx = 700.0
            self.fy = 700.0
            self.cx = w / 2
            self.cy = h / 2
            self.disto = [0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(self, w: int, h: int) -> None:
        self.left_cam = self.CamParams(w, h)
        self.right_cam = self.CamParams(w, h)
        self.stereo_transform = Transform()
        self.baseline = 0.12


class CameraConfiguration:
    def __init__(self, w: int, h: int) -> None:
        self.calibration_parameters = CalibrationParameters(w, h)
        self.resolution = type("R", (), {"width": w, "height": h})()


class CameraInformationFull:
    def __init__(self, w: int, h: int) -> None:
        self.camera_configuration = CameraConfiguration(w, h)


class Transform:
    def __init__(self) -> None:
        self.m = np.eye(4, dtype=np.float64)
        self.m[0, 3] = 0.12

    def get_translation(self):
        return type("T", (), {"get": lambda self=None: np.array([0.12, 0.0, 0.0])})()

    def get_rotation_matrix(self):
        return type("R", (), {"r": np.eye(3).flatten().tolist()})()


class Pose:
    def __init__(self) -> None:
        self.pose_data = type(
            "PD",
            (),
            {
                "translation": np.array([0.0, 0.0, 0.0]),
                "rotation": np.eye(3).flatten().tolist(),
            },
        )()


class PositionalTrackingParameters:
    def __init__(self) -> None:
        self.area_file = ""


class Camera:
    _frame_counter = 0

    def __init__(self) -> None:
        self._open = False
        self._tracking = False
        self._w, self._h = 1280, 720

    def open(self, init_params: InitParameters) -> ERROR_CODE:
        self._open = True
        if init_params.input is not None:
            self._w, self._h = 1280, 720
        return ERROR_CODE.SUCCESS

    def close(self) -> None:
        self._open = False

    def is_opened(self) -> bool:
        return self._open

    def grab(self, runtime_params) -> ERROR_CODE:
        if not self._open:
            return ERROR_CODE.SUCCESS
        Camera._frame_counter += 1
        if Camera._frame_counter > 200:
            return ERROR_CODE.END_OF_SVOFILE
        return ERROR_CODE.SUCCESS

    def retrieve_image(self, mat: Mat, view: VIEW, mem: MEM, resolution=None) -> ERROR_CODE:
        h, w = self._h, self._w
        if resolution is not None:
            w, h = resolution.width, resolution.height
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[..., 0] = 100 + Camera._frame_counter % 50
        bgra[..., 1] = 120
        bgra[..., 2] = 140
        bgra[..., 3] = 255
        mat.set_data(bgra)
        return ERROR_CODE.SUCCESS

    def retrieve_measure(self, *args, **kwargs):
        raise RuntimeError("retrieve_measure forbidden in mock tests unless guard disabled")

    def get_camera_information(self) -> CameraInformationFull:
        return CameraInformationFull(self._w, self._h)

    def get_camera_information_legacy(self):
        return CameraInformation()

    def enable_positional_tracking(self, params) -> ERROR_CODE:
        self._tracking = True
        return ERROR_CODE.SUCCESS

    def get_position(self, pose: Pose, ref) -> ERROR_CODE:
        t = Camera._frame_counter * 0.01
        pose.pose_data.translation = np.array([t, 0.0, 0.0])
        return ERROR_CODE.SUCCESS

    def get_timestamp(self, time_ref):
        class TS:
            @staticmethod
            def get_nanoseconds():
                return Camera._frame_counter * 33_000_000

        return TS()


def get_sdk_version() -> str:
    return "mock-zed-sdk-4.0"
