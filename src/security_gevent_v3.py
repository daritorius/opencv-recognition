# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import gevent
from gevent import monkey
monkey.patch_all()

import gc
import sys
import cv2
import numpy
import signal
import base64
import urllib3
import asyncio
import datetime
import argparse
import requests

from numba import jit
from time import sleep
from multiprocessing import cpu_count


"""
    Launch program with jemalloc installed:
        - MacOS:
            - jemalloc path: `/usr/local/Cellar/jemalloc/5.1.0/lib/libjemalloc.2.dylib`
            - run program: DYLD_INSERT_LIBRARIES={path to jamalloc} python security_gevent_v3.py
        - Linux:
            - jemalloc path: `/usr/local/lib/libjemalloc.so`
            - run program: LD_PRELOAD={path to jemalloc} python security_gevent_v3.py
"""


def init_parser():
    parser.add_argument(
        "--port",
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


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls._instances.get(cls, None) is None:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Security(object):
    __metaclass__ = Singleton
    __slots__ = (
        "loop",
        "debug",
        "api_key",
        "max_blur",
        "api_host",
        "cpu_count",
        "threshold",
        "time_delay",
        "start_array",
        "camera_port",
        "api_user_id",
        "movement_time",
        "current_array",
        "startup_count",
        "api_user_email",
        "detection_results",
        "max_startup_count",
        "api_request_timeout",
        "black_pixels_percent",

        "camera",
        "sensitivity",
        "camera_width",
        "camera_height",
        "camera_settings",
    )

    def __init__(self):
        # system config
        self.loop = None
        self.debug = False
        self.max_blur = 10
        self.cpu_count = cpu_count()
        self.threshold = 5
        self.time_delay = 15
        self.camera_port = 0
        self.start_array = None
        self.startup_count = 0
        self.current_array = None
        self.movement_time = None
        self.max_startup_count = 10
        self.detection_results = dict()
        self.api_request_timeout = 15
        self.black_pixels_percent = 70

        # API credentials
        self.api_host = "https://security.mybrains.org/"
        self.api_key = "eb2d06b672c81a0c5ce490840b4abf7082113c9efdb4ec66f944dc3f81f52b00370c23be68e32dbd23ea47e0aa2d7" \
                       "a748229d77180093fc102ebf8bd982bcc36FE9KuyBpl1D"
        self.api_user_id = 1
        self.api_user_email = "dm.sokoly@gmail.com"

        # camera config
        self.camera = None
        self.sensitivity = None
        self.camera_width = 1280
        self.camera_height = 720
        self.camera_settings = dict()

    def start(self):
        print("This script will use {} cores of your CPU to analyze video stream.".format(self.cpu_count))
        print("Parsing args...")

        try:
            # reset startup counter
            if self.startup_count:
                self.startup_count = 0

            self.init_camera()
        except ValueError as e:
            print(e)
            print("Restart in 5 seconds.")
            self.finish()
            sleep(5)
            return self.start()
        except KeyboardInterrupt:
            self.finish()
            print("Bye :) See you next time!")
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
            print("Restart in 5 seconds.")
            self.finish()
            sleep(5)
            return self.start()
        except KeyboardInterrupt:
            self.finish()
            print("Bye :) See you next time!")
            sys.exit(1)
        except Exception as e:
            if self.debug:
                import traceback
                print(traceback.format_exc())
            else:
                print(e)
            self.finish()
            print("Bye :) See you next time!")
            sys.exit(1)

    def motion_detection_callback(self, result):
        try:
            result.result()
        except ValueError:
            print("Restart in 5 seconds.")
            self.finish()
            sleep(5)
            return self.start()
        except Exception as e:
            print("Restart in 5 seconds.")
            self.finish()
            sleep(5)
            return self.start()
        except KeyboardInterrupt:
            self.finish()
            print("Bye :) See you next time!")
            sys.exit(1)

    async def start_security(self):
        print("Starting security process...")
        while True:
            self.current_array = self.capture_image()
            check_motion = self.check_motion(self.start_array, self.current_array)
            if check_motion:
                now = datetime.datetime.utcnow()
                if self.movement_time is None:
                    self.movement_time = now
                time_diff = int((now - self.movement_time).total_seconds() / 60)
                print("{}: Alert! Movement detected!".format(now.strftime("%Y-%m-%d %I:%M:%S %p")))
                if time_diff >= self.time_delay:
                    self.movement_time = now
                    gevent.joinall([gevent.spawn(self.process_detection, self.current_array)])
                    await asyncio.sleep(self.api_request_timeout)
            self.start_array = self.current_array
            await asyncio.sleep(1)

    def finish(self):
        if self.loop is not None:
            # cancel all tasks
            for task in asyncio.Task.all_tasks():
                try:
                    task.cancel()
                except asyncio.CancelledError:
                    print("Can't cancel task: {}.".format(task))
            self.loop.stop()
            self.loop.close()
            del self.loop
            self.loop = None

        gevent.signal(signal.SIGQUIT, gevent.kill)

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        # forcing garbage collector
        if len(gc.garbage):
            gc.collect()

    def init_camera(self):
        # init camera
        print("Starting camera....")

        if self.startup_count > self.max_startup_count:
            raise ValueError("Can't start camera")

        self.camera = cv2.VideoCapture(self.camera_port)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        sleep(5)

        if self.debug:
            self.get_camera_settings()

        print("Done.")
        print("Camera resolution is set to {}x{}".format(self.camera_width, self.camera_height))

        try:
            self.capture_initial_image()
        except ValueError as e:
            print(e)
            self.startup_count += 1
            sleep(5)
            return self.init_camera()

        self.sensitivity = int(self.camera_width * self.camera_height / self.cpu_count * 0.10)
        print("Camera sensitivity is set to: {}".format(self.sensitivity))

        self.start_array = self.current_array = self.capture_image()

    def get_camera_settings(self):
        self.camera_settings = dict(
            hue=self.camera.get(cv2.CAP_PROP_HUE),
            gain=self.camera.get(cv2.CAP_PROP_GAIN),
            test=self.camera.get(cv2.CAP_PROP_POS_MSEC),
            frame_rate=self.camera.get(cv2.CAP_PROP_FPS),
            exposure=self.camera.get(cv2.CAP_PROP_EXPOSURE),
            contrast=self.camera.get(cv2.CAP_PROP_CONTRAST),
            width=self.camera.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
            ratio=self.camera.get(cv2.CAP_PROP_POS_AVI_RATIO),
            brightness=self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
            saturation=self.camera.get(cv2.CAP_PROP_SATURATION),
        )
        print("Camera settings are set to: {}".format(self.camera_settings))

    def capture_initial_image(self, count=None, blur=None):
        if count is None:
            count = 1
        assert isinstance(count, int) and count >= 1

        assert isinstance(blur, type(None)) or isinstance(blur, float)

        if count > 5:
            if self.max_blur >= 10:
                self.max_blur -= 5
            else:
                self.max_blur = blur
            raise ValueError("Too many attempts to capture an initial image.\nRestart in 5 seconds.")

        # capture initial image
        print("Capturing initial image...")
        im = self.capture_image()
        if im is None:
            raise ValueError("Can't access to the camera :(")

        print("Expected blur ratio is: {}.".format(self.max_blur))

        # test blur rating
        blur = cv2.Laplacian(im, cv2.CV_64F).var()
        assert isinstance(blur, float)
        if blur < self.max_blur:
            print(
                "{}: Camera has lost focus. We can't analyze the initial picture. Blur rating is: {}.".format(
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %I:%M:%S %p"),
                    blur,
                )
            )
            sleep(2)
            return self.capture_initial_image(count=count+1, blur=blur)

        print("Blur rating is: {}.".format(blur))

        del im
        print("Done.")

    def capture_image(self):
        try:
            return self.camera.read()[1]
        except (AttributeError, IndexError):
            return None

    def count_black_pixels(self, array):
        assert isinstance(array, numpy.ndarray)
        total_pixels_count = self.camera_width * self.camera_height
        non_black_pixels_count = array.any(axis=-1).sum()
        return int((total_pixels_count - non_black_pixels_count) / total_pixels_count * 100)

    @jit(nogil=True)
    def test_images(self, i, data1, data2, start_index_width, end_index_width, start_index_height, end_index_height):
        assert isinstance(i, int) and i >= 0
        assert isinstance(data1, numpy.ndarray)
        assert isinstance(data2, numpy.ndarray)
        assert isinstance(start_index_width, int)
        assert isinstance(end_index_width, int)
        assert isinstance(start_index_height, int)
        assert isinstance(end_index_height, int)

        motion_detected = False
        pix_color = 1  # red=0 green=1 blue=2
        pix_changes = black_pixels = 0
        for w in range(start_index_width, end_index_width):
            for h in range(start_index_height, end_index_height):
                if data1[h][w].tolist() == [0, 0, 0]:
                    black_pixels += 1
                pix_diff = abs(int(data1[h][w][pix_color]) - int(data2[h][w][pix_color]))
                if pix_diff >= self.threshold:
                    pix_changes += 1
                if pix_changes >= self.sensitivity:
                    break
            if pix_changes >= self.sensitivity:
                break
        if pix_changes >= self.sensitivity:
            motion_detected = True

        w_diff = end_index_width - start_index_width
        h_diff = end_index_height - start_index_height
        check_black_pixels = black_pixels / w_diff * h_diff * 100
        if check_black_pixels >= self.black_pixels_percent:
            motion_detected = False

        self.detection_results[i] = motion_detected

        # forcing garbage collector
        if len(gc.garbage):
            gc.collect()

    @jit(nogil=True)
    def check_motion(self, array1, array2):
        if not isinstance(array1, numpy.ndarray) or not isinstance(array2, numpy.ndarray):
            raise ValueError("Expecting numpy array.")

        # check if array2 has many black pixels
        black_pixels = self.count_black_pixels(array2)
        assert isinstance(black_pixels, int)
        if black_pixels >= self.black_pixels_percent:
            return False

        # test blur rating
        blur = cv2.Laplacian(array2, cv2.CV_64F).var()
        assert isinstance(blur, float)
        if blur < self.max_blur:
            raise ValueError(
                "{}: Camera has lost focus. We can't analyze the initial picture. Blur rating is: {}.".format(
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %I:%M:%S %p"),
                    blur,
                )
            )

        tasks = []

        # reset detection results
        self.detection_results = dict()

        for i in range(0, self.cpu_count):
            start_index_width = self.camera_width / self.cpu_count * i
            end_index_width = (self.camera_width / self.cpu_count * (i + 1)) - 1
            start_index_height = self.camera_height / self.cpu_count * i
            end_index_height = (self.camera_height / self.cpu_count * (i + 1)) - 1
            tasks.append(
                gevent.spawn(
                    self.test_images,
                    i,
                    array1,
                    array2,
                    int(start_index_width),
                    int(end_index_width),
                    int(start_index_height),
                    int(end_index_height),
                )
            )

        gevent.joinall(tasks)
        gevent.signal(signal.SIGQUIT, gevent.kill)

        # forcing garbage collector
        if len(gc.garbage):
            gc.collect()

        return any(self.detection_results.values())

    def process_detection(self, array):
        assert isinstance(array, numpy.ndarray)
        if self.debug:
            print("Preparing image for sending...")
        image_string = self.get_image_string(array)
        if self.debug:
            print("Done")

        # send notification
        asyncio.ensure_future(self.send_notification(image_string), loop=self.loop)

    @staticmethod
    @jit(nogil=True)
    def get_image_string(array):
        assert isinstance(array, numpy.ndarray)
        return base64.b64encode(cv2.imencode('.jpg', array)[1])

    async def send_notification(self, image_string):
        assert isinstance(image_string, bytes)
        urllib3.disable_warnings(
            urllib3.exceptions.InsecureRequestWarning
        )
        data = {
            "user_id": self.api_user_id,
            "api_key": self.api_key,
            "email": self.api_user_email,
            "image": image_string.decode("utf-8"),
        }

        try:
            print("Sending notification...")
            response = requests.post(
                self.api_host,
                data=data,
                verify=False,
                headers={"Connection": "close"},
                timeout=self.api_request_timeout,
            )
            response.raise_for_status()
        except Exception:
            import traceback
            print(traceback.format_exc())
        else:
            print("Notification has been sent: {}".format(response.status_code))


if __name__ == "__main__":

    # init security object
    security = Security()

    # parse args
    parser = argparse.ArgumentParser(
        description="Process camera settings.",
    )
    init_parser()
    args = parser.parse_args()
    if args.port is not None:
        security.camera_port = args.port[0]
    if args.debug is not None:
        security.debug = False if not args.debug[0] else True
    if args.delay is not None:
        security.time_delay = args.delay[0]

    print("Done.")
    print("Setting up delay between messages to {} minutes.".format(security.time_delay))
    print("Done.")

    # start security
    security.start()
