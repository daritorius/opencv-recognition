# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import gevent
from gevent import monkey
monkey.patch_all()

import os
import cv2
import signal
import base64
import requests
import datetime
import argparse

from time import sleep
from multiprocessing import cpu_count


camera_port = 0
camera = None
clean = 0

image_width = 0
image_height = 0
threshold = 5
number_processes = cpu_count()
sensitivity = 0
black_pixels_percent = 85

api_host = "https://security.mybrains.org/"
user_api_key = "eb2d06b672c81a0c5ce490840b4abf7082113c9efdb4ec66f944dc3f81f52b00370c23be68e32dbd23ea47e0aa2d7a7482" \
               "29d77180093fc102ebf8bd982bcc36FE9KuyBpl1D"
user_id = 1
user_email = "dm.sokoly@gmail.com"
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
base_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'media/security'))

time_delay = 15

movement_time = None
debug = False

detection_results = dict()

camera_width = 1280
camera_height = 720


def get_camera_params():
    test = camera.get(cv2.CAP_PROP_POS_MSEC)
    ratio = camera.get(cv2.CAP_PROP_POS_AVI_RATIO)
    frame_rate = camera.get(cv2.CAP_PROP_FPS)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    brightness = camera.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = camera.get(cv2.CAP_PROP_CONTRAST)
    saturation = camera.get(cv2.CAP_PROP_SATURATION)
    hue = camera.get(cv2.CAP_PROP_HUE)
    gain = camera.get(cv2.CAP_PROP_GAIN)
    exposure = camera.get(cv2.CAP_PROP_EXPOSURE)
    print("Test: ", test)
    print("Ratio: ", ratio)
    print("Frame Rate: ", frame_rate)
    print("Height: ", height)
    print("Width: ", width)
    print("Brightness: ", brightness)
    print("Contrast: ", contrast)
    print("Saturation: ", saturation)
    print("Hue: ", hue)
    print("Gain: ", gain)
    print("Exposure: ", exposure)


def capture_image():
    im = camera.read()[1]
    return im


def read_image(image_media_path):
    image_array = cv2.imread(image_media_path)
    return image_array


def check_sensitivity(index, data1, data2, start_index_width, end_index_width, start_index_height,
                      end_index_height):
    motion_detected = False
    pix_color = 1  # red=0 green=1 blue=2
    pix_changes = black_pixels = 0
    for w in range(start_index_width, end_index_width):
        for h in range(start_index_height, end_index_height):
            if data1[h][w].tolist() == [0, 0, 0]:
                black_pixels += 1
            pix_diff = abs(int(data1[h][w][pix_color]) - int(data2[h][w][pix_color]))
            if pix_diff >= threshold:
                pix_changes += 1
            if pix_changes >= sensitivity:
                break
        if pix_changes >= sensitivity:
            break
    if pix_changes >= sensitivity:
        motion_detected = True
    check_black_pixels = black_pixels / (end_index_width - start_index_width) * \
                         (end_index_height - start_index_height) * 100
    if check_black_pixels >= black_pixels_percent:
        motion_detected = False
    detection_results[index] = motion_detected


def check_motion(array1, array2):
    tasks = []

    for i in range(0, number_processes):
        start_index_width = camera_width / number_processes * i
        end_index_width = (camera_width / number_processes * (i + 1)) - 1
        start_index_height = camera_height / number_processes * i
        end_index_height = (camera_height / number_processes * (i + 1)) - 1
        tasks.append(gevent.spawn(
            check_sensitivity,
            i,
            array1,
            array2,
            int(start_index_width),
            int(end_index_width),
            int(start_index_height),
            int(end_index_height),
        ))

    gevent.joinall(tasks)

    if debug:
        print(detection_results)

    return any(detection_results.values())


def get_image_string(array):
    return base64.b64encode(cv2.imencode('.jpg', array)[1])


def send_notification(image_string):
    data = {
        "user_id": user_id,
        "api_key": user_api_key,
        "email": user_email,
        "image": image_string.decode("utf-8"),
    }
    try:
        print("Sending notification...")
        response = requests.post(api_host, data=data, verify=False)
    except Exception:
        import traceback
        print(traceback.format_exc())
    else:
        print("Notification sent: %i" % response.status_code)


def remove_image(image_media_path):
    os.remove(image_media_path)


def process_detection(array):
    if debug:
        print("Preparing image for sending...")
    image_string = get_image_string(array)
    if debug:
        print("Done")
    send_notification(image_string)


if __name__ == "__main__":
    print("This script will use %i cores of your CPU to analyze video stream." % number_processes)
    print("Parsing args...")
    # parse args
    parser = argparse.ArgumentParser(description='Process camera settings.')
    parser.add_argument('--port', metavar='N', nargs='+', type=int,
                        help='Camera index in your system. Main camera is usually equal to 0.')
    parser.add_argument('--debug', metavar='N', nargs='+', type=int,
                        help='Debug settings 0|1, default is 0.')
    parser.add_argument('--delay', metavar='N', nargs='+', type=int,
                        help='Delay between notifications in minutes.')
    args = parser.parse_args()
    if args.port is not None:
        camera_port = args.port[0]
    if args.debug is not None:
        debug = False if not args.debug[0] else True
    if args.delay is not None:
        time_delay = args.delay[0]
    print("Done.")
    print("Setting up delay between messages to %i minutes." % time_delay)
    print("Done.")
    try:
        # init camera
        print("Starting camera....")
        camera = cv2.VideoCapture(camera_port)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        sleep(3)
        if debug:
            get_camera_params()
        print("Done.")
        print("Capturing initial image...")
        im = capture_image()
        print("Done.")
        print("Camera resolution is set to %ix%i" % (camera_width, camera_height))
        sensitivity = int(camera_width * camera_height / number_processes * 0.05)
        start_array = current_array = im
        print("Done.")
        print("Starting security process...")
        while True:
            startTime = datetime.datetime.utcnow()
            current_array = capture_image()
            if check_motion(start_array, current_array):
                if movement_time is None:
                    movement_time = datetime.datetime.utcnow()
                now = datetime.datetime.utcnow()
                time_diff = int((now - movement_time).total_seconds() / 60)
                local_now = datetime.datetime.now()
                print("%s: Alert! Movement detected!" % (local_now.strftime("%Y-%m-%d %I:%M:%S %p")))
                if time_diff >= time_delay:
                    movement_time = now
                    gevent.joinall([gevent.spawn(process_detection, current_array)])
                    sleep(1)
            im = capture_image()
            start_array = current_array
            if debug:
                print(datetime.datetime.utcnow() - startTime)
    except KeyboardInterrupt:
        camera.release()
        gevent.signal(signal.SIGQUIT, gevent.kill)
    except Exception as e:
        print(e)
        import traceback
        print(traceback.format_exc())
        camera.release()
        gevent.signal(signal.SIGQUIT, gevent.kill)
    finally:
        print('Bye :)')
