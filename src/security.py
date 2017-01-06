# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import random
import string
from multiprocessing import Process
from subprocess import call
from time import sleep
from PIL import Image

import cv2
import datetime
import glob

from multiprocessing import cpu_count, Manager


camera = None

image_width = 0
image_height = 0
threshold = 10
number_processes = cpu_count()
sensitivity = 0
black_pixels_percent = 80

api_host = "http://leksto.net/"
user_api_key = "eb2d06b672c81a0c5ce490840b4abf7082113c9efdb4ec66f944dc3f81f52b00370c23be68e32dbd23ea47e0aa2d7a7482" \
               "29d77180093fc102ebf8bd982bcc36FE9KuyBpl1D"
user_id = 1
user_email = "dm.sokoly@gmail.com"
base_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'media/security'))


def get_camera_params():
    test = camera.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    ratio = camera.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)
    frame_rate = camera.get(cv2.cv.CV_CAP_PROP_FPS)
    width = camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    brightness = camera.get(cv2.cv.CV_CAP_PROP_BRIGHTNESS)
    contrast = camera.get(cv2.cv.CV_CAP_PROP_CONTRAST)
    saturation = camera.get(cv2.cv.CV_CAP_PROP_SATURATION)
    hue = camera.get(cv2.cv.CV_CAP_PROP_HUE)
    gain = camera.get(cv2.cv.CV_CAP_PROP_GAIN)
    exposure = camera.get(cv2.cv.CV_CAP_PROP_EXPOSURE)
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
    retval, im = camera.read()
    random_part = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(11))
    now = datetime.datetime.utcnow()
    image_time = now.strftime("%Y%m%d_%H%M%S")
    image_media_path = "media/security/capture_%s_%s.jpg" % (random_part, image_time)
    cv2.imwrite(image_media_path, im)
    return im, image_media_path


def read_image(image_media_path):
    image_array = cv2.imread(image_media_path)
    return image_array


def get_image_resolution(image_media_path):
    with Image.open(image_media_path) as im:
        width, height = im.size
    return width, height


def check_sensitivity(index, data1, data2, start_index_width, end_index_width, start_index_height,
                      end_index_height, results):
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
    results[index] = motion_detected


def check_motion(array1, array2):
    manager = Manager()
    results = manager.dict()

    for i in range(0, number_processes):
        if i == 0:
            start_index_width = start_index_height = 0
            end_index_width = image_width / number_processes
            end_index_height = image_height / number_processes
        else:
            start_index_width = image_width / number_processes * i + 1
            start_index_height = image_height / number_processes * i + 1
            end_index_width = image_width / number_processes * (i + 1)
            end_index_height = image_height / number_processes * (i + 1)
        p = Process(
            target=check_sensitivity,
            args=(
                i,
                array1,
                array2,
                start_index_width,
                end_index_width,
                start_index_height,
                end_index_height,
                results
            )
        )
        p.daemon = True
        p.start()
        p.join()

    motion_detected = False
    for value in results.values():
        if value:
            motion_detected = value
            break
    return motion_detected


def remove_image(image_media_path):
    os.remove(image_media_path)


def clean_images():
    files = glob.glob("%s/*.jpg" % base_image_path)
    for dump in files:
        call(["rm", "-rf", dump])


if __name__ == "__main__":
    try:
        clean_images()
        camera = cv2.VideoCapture(0)
        sleep(3)
        get_camera_params()
        im, image_media_path = capture_image()
        image_resolution = get_image_resolution(image_media_path)
        image_width = image_resolution[0]
        image_height = image_resolution[1]
        sensitivity = int(image_width * image_height / number_processes * 0.10)
        start_array = current_array = read_image(image_media_path)
        while True:
            current_array = read_image(image_media_path)
            if check_motion(start_array, current_array):
                print("Movement")
                remove_image(image_media_path)
                im, image_media_path = capture_image()
                start_array = current_array
                continue
            remove_image(image_media_path)
            im, image_media_path = capture_image()
            start_array = current_array
    except KeyboardInterrupt:
        camera.release()
        del (camera)
    finally:
        print('Bye')
