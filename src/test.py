# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import cv
import random
import string

if __name__ == "__main__":
    capture = cv.CaptureFromCAM(1)
    frame = cv.QueryFrame(capture)
    random_part = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(11))
    cv.SaveImage("media/capture_%s.jpg" % random_part, frame)
