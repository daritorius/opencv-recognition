# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import gevent

from gevent import monkey

monkey.patch_all()

import gc
import sys
import cv2
import zlib
import json
import time
import numpy
import signal
import base64
import pickle
import asyncio
import datetime
import argparse
import requests

from time import sleep
from multiprocessing import cpu_count


"""
    Launch program with jemalloc installed:
        - MacOS:
            - jemalloc path: `/usr/local/Cellar/jemalloc/5.1.0/lib/libjemalloc.2.dylib`
            - run program: DYLD_INSERT_LIBRARIES={path to jamalloc} python -u -B security_gevent_v3.py
        - Linux:
            - jemalloc path: `/usr/local/lib/libjemalloc.so`
            - run program: LD_PRELOAD={path to jemalloc} python -u -B security_gevent_v3.py
"""


def init_parser():
    parser.add_argument(
        "--camera",
        metavar="N",
        nargs="+",
        type=int,
        help="Camera index in your system. Main camera is usually equal to 0.",
    )
    parser.add_argument(
        "--debug",
        metavar="N",
        nargs="+",
        type=int,
        help="Debug settings 0|1, default is 0.",
    )
    parser.add_argument(
        "--delay",
        metavar="N",
        nargs="+",
        type=int,
        help="Delay between notifications in minutes.",
    )
    parser.add_argument(
        "--video",
        metavar="N",
        nargs="+",
        type=int,
        help="Capture video.",
    )
    parser.add_argument(
        "--host",
        metavar="S",
        nargs="+",
        type=str,
        help="Host of security HUB.",
    )
    parser.add_argument(
        "--port",
        metavar="S",
        nargs="+",
        type=str,
        help="Port of security HUB.",
    )


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls._instances.get(cls, None) is None:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Security(object, metaclass=Singleton):
    __slots__ = (
        # system
        "loop",
        "debug",
        "max_blur",
        "cpu_count",
        "time_delay",
        "start_array",
        "blur_values",
        "camera_port",
        "startup_count",
        "current_array",
        "movement_time",
        "default_timeout",
        "detection_results",
        "max_startup_count",
        "api_request_timeout",
        "black_pixels_percent",
        "white_pixels_percent",

        # api
        "api_key",
        "api_host",
        "api_user_id",
        "api_user_email",

        # hub
        "hub_host",
        "hub_port",

        # camera
        "sens",
        "camera",
        "threshold",
        "threshold_p",
        "sensitivity",
        "camera_width",
        "camera_height",
        "camera_hq_fps",
        "camera_settings",
        "camera_normal_fps",
        "camera_detect_fps",
        "camera_video_width",
        "camera_video_height",
        "camera_video_length",
        "camera_detect_width",
        "camera_detect_height",
        "camera_capture_video",
        "max_camera_reload_count",
    )

    def __init__(self):
        # system config
        self.loop = None
        self.debug = False
        self.max_blur = 1000
        self.cpu_count = cpu_count()
        self.time_delay = 15
        self.blur_values = []
        self.camera_port = 0
        self.start_array = None
        self.startup_count = 0
        self.current_array = None
        self.movement_time = None
        self.max_startup_count = 10
        self.default_timeout = 2
        self.detection_results = dict()
        self.api_request_timeout = 20
        self.black_pixels_percent = 80
        self.white_pixels_percent = 80

        # API credentials
        self.api_host = ""
        self.api_key = ""
        self.api_user_id = None
        self.api_user_email = ""

        # HUB
        self.hub_host = "127.0.0.1"
        self.hub_port = "8000"

        # camera config
        self.sens = 0.09
        self.camera = None
        self.threshold = None
        self.threshold_p = 0.005
        self.sensitivity = None
        self.camera_width = 1280
        self.camera_height = 720
        self.camera_hq_fps = 60
        self.camera_settings = dict()
        self.camera_detect_fps = 10
        self.camera_normal_fps = 30
        self.camera_video_width = 640
        self.camera_video_height = 480
        self.camera_video_length = 10
        self.camera_detect_width = 80
        self.camera_detect_height = 40
        self.camera_capture_video = False
        self.max_camera_reload_count = 3

        print("[UTC: {}] This program will use {} cores of your CPU to analyze video stream.".format(
            self.get_now_date(), self.cpu_count,
        ))
        print("[UTC: {}] Parsing args...".format(self.get_now_date()))
        print("[UTC: {}] The initial security object size is {} bytes.".format(
            self.get_now_date(),
            sys.getsizeof(self),
        ))

    @staticmethod
    def get_now_date():
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %I:%M:%S.%f %p")

    def readjust_blur(self):
        # set max blur based on current image
        blur = cv2.Laplacian(self.current_array, cv2.CV_64F).var()
        assert isinstance(blur, float)
        self.max_blur = blur - (blur / 15.0)
        print("[UTC: {}] Blur ratio has been updated to: {}.".format(self.get_now_date(), self.max_blur))

    def start(self):
        try:
            # reset startup counter
            if self.startup_count:
                self.startup_count = 0

            self.init_camera()
        except ValueError as e:
            print(e)
            print("[UTC: {}] Restart in {} seconds.".format(self.get_now_date(), self.default_timeout))
            self.finish()
            sleep(self.default_timeout)
            return self.start()
        except KeyboardInterrupt:
            self.finish()
            print("[UTC: {}] Bye :) See you next time!".format(self.get_now_date()))
            sys.exit(1)

        try:
            self.loop = asyncio.get_event_loop()
            if self.loop.is_closed() or self.loop.is_running():
                if self.loop.is_running():
                    self.loop.stop()
                self.loop = asyncio.new_event_loop()
            _f = asyncio.ensure_future(self.start_security(), loop=self.loop)
            self.loop.run_until_complete(_f)
            _f.add_done_callback(self.motion_detection_callback)
        except ValueError as e:
            print(e)
            print("[UTC: {}] Restart in {} seconds.".format(self.get_now_date(), self.default_timeout))
            self.finish()
            sleep(self.default_timeout)
            return self.start()
        except KeyboardInterrupt:
            self.finish()
            print("[UTC: {}] Bye :) See you next time!".format(self.get_now_date()))
            sys.exit(1)
        except Exception as e:
            if self.debug:
                import traceback
                print(traceback.format_exc())
            else:
                print(e)
            self.finish()
            print("[UTC: {}] Bye :) See you next time!".format(self.get_now_date()))
            sys.exit(1)

    def motion_detection_callback(self, result):
        try:
            result.result()
        except ValueError:
            print("[UTC: {}] Restart in {} seconds.".format(self.get_now_date(), self.default_timeout))
            self.finish()
            sleep(self.default_timeout)
            return self.start()
        except Exception as e:
            print("[UTC: {}] Restart in {} seconds.".format(self.get_now_date(), self.default_timeout))
            self.finish()
            sleep(self.default_timeout)
            return self.start()
        except KeyboardInterrupt:
            self.finish()
            print("[UTC: {}] Bye :) See you next time!".format(self.get_now_date()))
            sys.exit(1)

    async def start_security(self):
        # reset blur values
        self.blur_values = []

        print("[UTC: {}] Starting security process...".format(self.get_now_date()))
        while True:
            self.current_array = self.capture_image()
            check_motion = self.check_motion(self.start_array, self.current_array)
            assert isinstance(check_motion, bool)
            if check_motion:
                now = datetime.datetime.utcnow()
                if self.movement_time is None:
                    self.movement_time = now
                time_diff = int((now - self.movement_time).total_seconds() / 60)
                print(
                    "[UTC: {}] Movement detected!".format(
                        self.get_now_date(),
                    )
                )
                if time_diff >= self.time_delay:
                    self.movement_time = now

                    # capture security video
                    if self.camera_capture_video:
                        self.capture_video()

            self.start_array = self.current_array
            # await asyncio.sleep(.05)

    def finish(self):
        if self.loop is not None:
            # cancel all tasks
            for task in asyncio.Task.all_tasks():
                try:
                    task.cancel()
                except asyncio.CancelledError:
                    print("[UTC: {}] Can't cancel task: {}.".format(self.get_now_date(), task))
            self.loop.stop()
            self.loop.close()
            del self.loop
            self.loop = None

        gevent.signal(signal.SIGQUIT, gevent.kill)

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        # remove all windows, opened by cv2
        cv2.destroyAllWindows()

        # forcing garbage collector
        if len(gc.garbage):
            gc.collect()

    def init_camera(self):
        # init camera
        print("[UTC: {}] Starting camera....".format(self.get_now_date()))

        if self.startup_count > self.max_startup_count:
            raise ValueError("[UTC: {}] Can't start camera".format(self.get_now_date()))

        self.camera = cv2.VideoCapture(self.camera_port)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.set_camera_detect_resolution()
        self.get_camera_settings()
        sleep(self.default_timeout)

        print("[UTC: {}] Done.".format(self.get_now_date()))

        try:
            self.capture_initial_image()
        except ValueError as e:
            print(e)
            self.startup_count += 1
            self.finish()
            sleep(self.default_timeout)
            return self.init_camera()

        self.threshold = int(self.camera_detect_width * self.camera_detect_height / self.cpu_count * self.threshold_p)
        self.sensitivity = int(self.camera_detect_width * self.camera_detect_height / self.cpu_count * self.sens)
        print("[UTC: {}] Camera sensitivity is set to {} pixels out of {} per frame.".format(
            self.get_now_date(),
            self.sensitivity,
            int(self.camera_detect_width * self.camera_detect_height / self.cpu_count),
        ))

        self.start_array = self.current_array = self.capture_image()

    def set_camera_full_resolution(self):
        self.camera.set(cv2.CAP_PROP_FPS, self.camera_normal_fps)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

    def set_camera_detect_resolution(self):
        self.camera.set(cv2.CAP_PROP_FPS, self.camera_detect_fps)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_detect_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_detect_height)

    def set_camera_video_resolution(self):
        self.camera.set(cv2.CAP_PROP_FPS, self.camera_normal_fps)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_video_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_video_height)

    def get_camera_settings(self):
        self.camera_settings = dict(
            hue=self.camera.get(cv2.CAP_PROP_HUE),
            fps=self.camera.get(cv2.CAP_PROP_FPS),
            gain=self.camera.get(cv2.CAP_PROP_GAIN),
            test=self.camera.get(cv2.CAP_PROP_POS_MSEC),
            exposure=self.camera.get(cv2.CAP_PROP_EXPOSURE),
            contrast=self.camera.get(cv2.CAP_PROP_CONTRAST),
            width=self.camera.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
            ratio=self.camera.get(cv2.CAP_PROP_POS_AVI_RATIO),
            brightness=self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
            saturation=self.camera.get(cv2.CAP_PROP_SATURATION),
        )
        print("[UTC: {}] Camera settings are set to: {}".format(self.get_now_date(), self.camera_settings))

    def capture_initial_image(self, count=None, blur=None):
        if count is None:
            count = 1
        assert isinstance(count, int) and count >= 1

        assert isinstance(blur, type(None)) or isinstance(blur, float)

        if count > self.max_camera_reload_count:
            self.max_blur = min(self.blur_values)
            raise ValueError(
                "[UTC: {}] Too many attempts to capture an initial image. Restart in {} seconds.".format(
                    self.get_now_date(),
                    self.default_timeout,
                )
            )

        # capture initial image
        print("[UTC: {}] Capturing initial image...".format(self.get_now_date()))
        im = self.capture_image()
        if im is None:
            raise ValueError("[UTC: {}] Can't access to the camera :(".format(self.get_now_date()))

        # test blur rating
        blur = cv2.Laplacian(im, cv2.CV_64F).var()
        self.blur_values.append(blur)
        assert isinstance(blur, float)
        if blur < self.max_blur:
            print(
                "[UTC: {}] Camera has lost focus. We can't analyze the initial picture. Blur rating is: {}.".format(
                    self.get_now_date(),
                    blur,
                )
            )
            sleep(self.default_timeout)
            return self.capture_initial_image(count=count+1, blur=blur)

        self.max_blur = min(self.blur_values)
        print("[UTC: {}] Blur rating is: {}.".format(self.get_now_date(), self.max_blur))

        del im
        print("[UTC: {}] Done.".format(self.get_now_date()))

    def capture_image(self):
        try:
            img = self.camera.read()[1]
            # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        except (AttributeError, IndexError):
            return None

    def check_motion(self, array1, array2):
        if not isinstance(array1, numpy.ndarray) or not isinstance(array2, numpy.ndarray):
            raise ValueError("[UTC: {}] Expecting numpy array.".format(self.get_now_date()))

        start = time.time()

        try:
            response = requests.post(
                "http://{host}:{port}".format(host=self.hub_host, port=self.hub_port),
                data={
                    "image1": base64.b64encode(zlib.compress(pickle.dumps(array1, protocol=0))),
                    "image2": base64.b64encode(zlib.compress(pickle.dumps(array2, protocol=0))),
                },
                verify=False,
                headers={"Connection": "close"},
                timeout=self.api_request_timeout,
            )
        except Exception:
            raise ValueError("HUB is down.")
        else:
            if not response.ok:
                raise ValueError("HUB is down.")

            if len(response.json()["errors"]):
                for e in response.json()["errors"]:
                    if e.get("code") is not None and e["code"] == 100:
                        blur = cv2.Laplacian(array2, cv2.CV_64F).var()
                        self.blur_values.append(blur)
                        assert isinstance(blur, float)
                    elif e.get("code") is not None and e["code"] == 101:
                        blur = cv2.Laplacian(array2, cv2.CV_64F).var()
                        self.blur_values.append(blur)
                        assert isinstance(blur, float)
                raise ValueError(response.json()["errors"])

            end = time.time()
            print("Checked motion in {} ms".format((end - start) * 1000))
            return response.json()["data"]["is_motion_detected"]

    def capture_video(self):
        # set video settings for camera
        self.set_camera_video_resolution()

        # activate decoder for video
        four_cc = cv2.VideoWriter_fourcc(*"H264")

        out_file = cv2.VideoWriter(
            "output.mp4",
            four_cc,
            self.camera_normal_fps,
            (
                self.camera_video_width,
                self.camera_video_height,
            ),
        )

        end_video_at = datetime.datetime.utcnow() + datetime.timedelta(seconds=self.camera_video_length)
        while True:
            _camera_available, frame = self.camera.read()

            if _camera_available:
                out_file.write(frame)
            else:
                break

            if datetime.datetime.utcnow() > end_video_at:
                break

        # TODO: send frames to security host

        # close file
        out_file.release()

        # set detection settings for camera
        self.set_camera_detect_resolution()


if __name__ == "__main__":

    # init security object
    security = Security()

    # parse args
    parser = argparse.ArgumentParser(
        description="Process camera settings.",
    )
    init_parser()
    args = parser.parse_args()
    if args.camera is not None:
        security.camera_port = args.camera[0]
    if args.debug is not None:
        security.debug = False if not args.debug[0] else True
    if args.delay is not None:
        security.time_delay = args.delay[0]
    if args.video is not None:
        security.camera_capture_video = False if not args.video[0] else True
    if args.host is not None:
        security.hub_host = args.host[0]
    if args.port is not None:
        security.hub_port = args.port[0]

    print("[UTC: {}] Done.".format(security.get_now_date()))
    print("[UTC: {}] Setting up delay between messages to {} minutes.".format(
        security.get_now_date(), security.time_delay
    ))
    print("[UTC: {}] Done.".format(security.get_now_date()))

    # start security
    security.start()
