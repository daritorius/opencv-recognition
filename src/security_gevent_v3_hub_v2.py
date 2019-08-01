# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pickle

import gevent
from gevent import monkey

monkey.patch_all()

import gc
import sys
import cv2
import zlib
import time
import numpy
import signal
import base64
import asyncio
import datetime
import argparse
import requests

from time import sleep


"""
    Launch program with jemalloc installed:
        - MacOS:
            - jemalloc path: `/usr/local/Cellar/jemalloc/5.2.0/lib/libjemalloc.2.dylib`
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


class Security(metaclass=Singleton):

    __slots__ = (
        # system
        "loop",
        "debug",
        "camera_port",
        "current_array",
        "default_timeout",
        "api_request_timeout",

        # api
        "api_key",
        "api_host",
        "api_user_id",
        "api_user_email",

        # hub
        "uid",
        "hub_host",
        "hub_port",

        # camera
        "camera",
        "camera_width",
        "camera_height",
        "camera_hq_fps",
        "camera_settings",
        "camera_normal_fps",
        "camera_detect_fps",
        "camera_detect_width",
        "camera_detect_height",
    )

    def __init__(self):
        # system config
        self.loop = None
        self.debug = False
        self.camera_port = 0
        self.current_array = None
        self.default_timeout = 3
        self.api_request_timeout = 15

        # API credentials
        self.api_host = ""
        self.api_key = ""
        self.api_user_id = None
        self.api_user_email = ""

        # HUB
        self.uid = None
        self.hub_host = "127.0.0.1"
        self.hub_port = "8000"

        # camera config
        self.camera = None
        self.camera_width = 1280
        self.camera_height = 720
        self.camera_hq_fps = 10
        self.camera_settings = dict()
        self.camera_detect_fps = 10
        self.camera_normal_fps = 10
        self.camera_detect_width = 80
        self.camera_detect_height = 40

        print("[{}] Parsing args...".format(self.get_now_date()))
        print("[{}] The initial security object size is {} bytes.".format(
            self.get_now_date(),
            sys.getsizeof(self),
        ))

    @staticmethod
    def get_now_date():
        return "UTC: {}".format(datetime.datetime.utcnow().strftime("%Y-%m-%d %I:%M:%S.%f %p"))

    def start(self):
        try:
            self.init_camera()
        except ValueError as e:
            print(e)
            print("[{}] Restart in {} seconds.".format(self.get_now_date(), self.default_timeout))
            self.finish()
            sleep(self.default_timeout)
            return self.start()
        except KeyboardInterrupt:
            self.finish()
            print("[{}] Bye :) See you next time!".format(self.get_now_date()))
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
            print("[{}] Restart in {} seconds.".format(self.get_now_date(), self.default_timeout))
            self.finish()
            sleep(self.default_timeout)
            return self.start()
        except KeyboardInterrupt:
            self.finish()
            print("[{}] Bye :) See you next time!".format(self.get_now_date()))
            sys.exit(1)
        except Exception as e:
            if self.debug:
                import traceback
                print(traceback.format_exc())
            else:
                print(e)
            self.finish()
            print("[{}] Bye :) See you next time!".format(self.get_now_date()))
            sys.exit(1)

    def motion_detection_callback(self, result):
        try:
            result.result()
        except ValueError:
            print("[{}] Restart in {} seconds.".format(self.get_now_date(), self.default_timeout))
            self.finish()
            sleep(self.default_timeout)
            return self.start()
        except Exception as e:
            print("[{}] Restart in {} seconds.".format(self.get_now_date(), self.default_timeout))
            self.finish()
            sleep(self.default_timeout)
            return self.start()
        except KeyboardInterrupt:
            self.finish()
            print("[{}] Bye :) See you next time!".format(self.get_now_date()))
            sys.exit(1)

    async def start_security(self):
        print("[{}] Starting security process...".format(self.get_now_date()))
        while True:
            self.current_array = self.capture_image()
            cm = self.check_motion(self.current_array)
            assert isinstance(cm, bool)
            if cm:
                print(
                    "[{}] Movement detected!".format(
                        self.get_now_date(),
                    )
                )

    def finish(self):
        if self.loop is not None:
            # cancel all tasks
            for task in asyncio.Task.all_tasks():
                try:
                    task.cancel()
                except asyncio.CancelledError:
                    print("[{}] Can't cancel task: {}.".format(self.get_now_date(), task))
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
        print("[{}] Starting camera....".format(self.get_now_date()))

        self.camera = cv2.VideoCapture(self.camera_port)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        self.camera_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.camera_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(
            "[{}] Camera initial resolution is: {}x{}".format(
                self.get_now_date(), self.camera_width, self.camera_height
            )
        )

        self.set_camera_detect_resolution()
        sleep(self.default_timeout)

        print("[{}] Done.".format(self.get_now_date()))

    def set_camera_full_resolution(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.get_camera_settings()

    def set_camera_detect_resolution(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_detect_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_detect_height)
        self.get_camera_settings()

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
        print("[{}] Camera settings are set to: {}".format(self.get_now_date(), self.camera_settings))

    def capture_image(self):
        try:
            img = self.camera.read()[1]
            return img
        except (AttributeError, IndexError):
            return None

    def check_motion(self, array):
        if not isinstance(array, numpy.ndarray):
            raise ValueError("[{}] Expecting numpy array.".format(self.get_now_date()))

        start = time.time()

        try:
            r_data = {
                "image": base64.b64encode(zlib.compress(pickle.dumps(array, protocol=0))),
            }
            if self.uid is not None:
                r_data["uid"] = self.uid
            response = requests.post(
                "http://{host}:{port}/v2".format(host=self.hub_host, port=self.hub_port),
                data=r_data,
                verify=False,
                headers={"Connection": "close"},
                timeout=self.api_request_timeout,
            )
            response.raise_for_status()
        except Exception:
            raise ValueError("HUB is down.")
        else:
            if not response.ok:
                raise ValueError("HUB is down.")

            r = response.json()

            if r["data"].get("uid") is not None and self.uid is None:
                print("[{}] Camera UID is {}".format(self.get_now_date(), r["data"]["uid"]))
                self.uid = r["data"]["uid"]

            if r["data"]["is_notification_needed"]:
                self.send_full_resolution_image()

            if len(r["errors"]):
                raise ValueError(r["errors"])

            end = time.time()
            if self.debug:
                print("[{}] Checked motion in {} ms".format(self.get_now_date, (end - start) * 1000))
            return r["data"]["is_motion_detected"]

    def send_full_resolution_image(self):
        # set full resolution to camera
        self.set_camera_full_resolution()

        # capture image
        image = self.capture_image()

        # set detect resolution
        self.set_camera_detect_resolution()

        if not isinstance(image, numpy.ndarray):
            raise ValueError("[{}] Expecting numpy array.".format(self.get_now_date()))

        print("[{}] Sending full size picture.".format(self.get_now_date()))
        try:
            r_data = {
                "image": base64.b64encode(zlib.compress(pickle.dumps(image, protocol=0))),
            }
            if self.uid is not None:
                r_data["uid"] = self.uid
            response = requests.post(
                "http://{host}:{port}/notification".format(host=self.hub_host, port=self.hub_port),
                data=r_data,
                verify=False,
                headers={"Connection": "close"},
                timeout=self.api_request_timeout,
            )
            response.raise_for_status()
        except Exception:
            raise ValueError("HUB is down.")


if __name__ == "__main__":

    # init security object
    security = Security()

    # parse args
    parser = argparse.ArgumentParser(
        description="[{}] Process camera settings.".format(security.get_now_date()),
    )
    init_parser()
    args = parser.parse_args()
    if args.camera is not None:
        security.camera_port = args.camera[0]
    if args.debug is not None:
        security.debug = False if not args.debug[0] else True
    if args.host is not None:
        security.hub_host = args.host[0]
    if args.port is not None:
        security.hub_port = args.port[0]

    # start security
    security.start()
