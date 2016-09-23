# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import cv


if __name__ == "__main__":
    capture = cv.CaptureFromCAM(1)
    frame = cv.QueryFrame(capture)
    cv.SaveImage("media/capture.jpg", frame)
