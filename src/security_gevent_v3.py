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
            - run program: DYLD_INSERT_LIBRARIES={path to jamalloc} python -u -B security_gevent_v3.py
        - Linux:
            - jemalloc path: `/usr/local/lib/libjemalloc.so`
            - run program: LD_PRELOAD={path to jemalloc} python -u -B security_gevent_v3.py
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
    parser.add_argument(
        "--video",
        metavar="N",
        nargs="+",
        type=int,
        help="Capture video.",
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
        self.api_request_timeout = 15
        self.black_pixels_percent = 70
        self.white_pixels_percent = 80

        # API credentials
        self.api_host = "https://security.mybrains.org/"
        self.api_key = "eb2d06b672c81a0c5ce490840b4abf7082113c9efdb4ec66f944dc3f81f52b00370c23be68e32dbd23ea47e0aa2d7" \
                       "a748229d77180093fc102ebf8bd982bcc36FE9KuyBpl1D"
        self.api_user_id = 1
        self.api_user_email = "dm.sokoly@gmail.com"

        # camera config
        self.sens = 0.1
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
        self.camera_detect_width = 160
        self.camera_detect_height = 90
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

                    # process movement
                    gevent.joinall([gevent.spawn(self.process_detection)])
                    await asyncio.sleep(self.api_request_timeout)

                    # capture security video
                    if self.camera_capture_video:
                        self.capture_video()

            self.start_array = self.current_array
            await asyncio.sleep(1)

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
                "[UTC: {}] Too many attempts to capture an initial image.\nRestart in {} seconds.".format(
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
            return self.camera.read()[1]
        except (AttributeError, IndexError):
            return None

    def count_black_pixels(self, array):
        assert isinstance(array, numpy.ndarray)

        # get total pixels per frame
        total_pixels_count = self.camera_detect_width * self.camera_detect_height

        # calculate black pixels per frame
        min_color_range = numpy.array([0, 0, 0], numpy.uint8)
        max_color_range = numpy.array([25, 25, 25], numpy.uint8)
        black_pixels_count = cv2.countNonZero(cv2.inRange(array, min_color_range, max_color_range))
        return int(black_pixels_count / total_pixels_count * 100)

    def count_white_pixels(self, array):
        assert isinstance(array, numpy.ndarray)

        # get total pixels per frame
        total_pixels_count = self.camera_detect_width * self.camera_detect_height

        # calculate black pixels per frame
        min_color_range = numpy.array([250, 250, 250], numpy.uint8)
        max_color_range = numpy.array([255, 255, 255], numpy.uint8)
        white_pixels_count = cv2.countNonZero(cv2.inRange(array, min_color_range, max_color_range))
        return int(white_pixels_count / total_pixels_count * 100)

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
        pix_changes = 0
        for w in range(start_index_width, end_index_width):
            for h in range(start_index_height, end_index_height):
                pix_diff = abs(int(data1[h][w][pix_color]) - int(data2[h][w][pix_color]))
                if pix_diff >= self.threshold:
                    pix_changes += 1
                if pix_changes >= self.sensitivity:
                    break
            if pix_changes >= self.sensitivity:
                break
        if pix_changes >= self.sensitivity:
            motion_detected = True

        self.detection_results[i] = motion_detected

        # forcing garbage collector
        if len(gc.garbage):
            gc.collect()

    @jit(nogil=True)
    def check_motion(self, array1, array2):
        if not isinstance(array1, numpy.ndarray) or not isinstance(array2, numpy.ndarray):
            raise ValueError("[UTC: {}] Expecting numpy array.".format(self.get_now_date()))

        # check if array2 has many black pixels
        black_pixels = self.count_black_pixels(array2)
        assert isinstance(black_pixels, int)
        if black_pixels >= self.black_pixels_percent:
            print(
                "[UTC: {}] Frame contains too many black pixels: {}% (max {}%).".format(
                    self.get_now_date(),
                    black_pixels,
                    self.black_pixels_percent,
                )
            )
            return False

        # check if array2 has many white pixels
        white_pixels = self.count_white_pixels(array2)
        assert isinstance(white_pixels, int)
        if white_pixels >= self.white_pixels_percent:
            print(
                "[UTC: {}] Frame contains too many white pixels: {}% (max {}%).".format(
                    self.get_now_date(),
                    white_pixels,
                    self.white_pixels_percent,
                )
            )
            return False

        # test blur rating
        blur = cv2.Laplacian(array2, cv2.CV_64F).var()
        assert isinstance(blur, float)
        if blur < self.max_blur:
            self.readjust_blur()

        tasks = []

        # reset detection results
        self.detection_results = dict()

        for i in range(0, self.cpu_count):
            start_index_width = self.camera_detect_width / self.cpu_count * i
            end_index_width = (self.camera_detect_width / self.cpu_count * (i + 1)) - 1
            start_index_height = self.camera_detect_height / self.cpu_count * i
            end_index_height = (self.camera_detect_height / self.cpu_count * (i + 1)) - 1
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

    def process_detection(self):
        # set full resolution for camera
        self.set_camera_full_resolution()

        # capture image
        array = self.capture_image()
        assert isinstance(array, numpy.ndarray)
        if self.debug:
            print("[UTC: {}] Preparing image for sending...".format(self.get_now_date()))

        image_string = self.get_image_string(array)
        if self.debug:
            print("[UTC: {}] Done.".format(self.get_now_date()))

        # send notification
        asyncio.ensure_future(self.send_notification(image_string), loop=self.loop)

        # revert camera resolution to detect mode
        self.set_camera_detect_resolution()

        # set max blur based on current image
        blur = cv2.Laplacian(self.current_array, cv2.CV_64F).var()
        assert isinstance(blur, float)
        self.max_blur = blur - (blur / 15.0)
        print("[UTC: {}] Blur ratio has been updated to: {}.".format(self.get_now_date(), self.max_blur))
        print("[UTC: {}] Security object size is {} bytes.".format(self.get_now_date(), sys.getsizeof(self)))

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

    @staticmethod
    @jit(nogil=True)
    def get_image_string(array):
        assert isinstance(array, numpy.ndarray)
        return base64.b64encode(cv2.imencode('.jpg', array)[1])

    async def send_notification(self, image_string):
        assert isinstance(image_string, bytes)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        data = {
            "user_id": self.api_user_id,
            "api_key": self.api_key,
            "email": self.api_user_email,
            "image": image_string.decode("utf-8"),
        }

        try:
            print("[UTC: {}] Sending notification...".format(self.get_now_date()))
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
            print("[UTC: {}] Notification has been sent: {}".format(self.get_now_date(), response.status_code))


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
    if args.video is not None:
        security.camera_capture_video = False if not args.video[0] else True

    print("[UTC: {}] Done.".format(security.get_now_date()))
    print("[UTC: {}] Setting up delay between messages to {} minutes.".format(
        security.get_now_date(), security.time_delay
    ))
    print("[UTC: {}] Done.".format(security.get_now_date()))

    # start security
    security.start()
