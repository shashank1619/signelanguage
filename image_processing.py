import numpy as np
import cv2

minValue = 70


def func(path):
    frame = cv2.imread(path)

    # Save original image
    # cv2.imwrite("0original_image.jpg", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("1gray_image.jpg", gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # cv2.imwrite("2blurred_image.jpg", blur)

    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    # cv2.imwrite("3adaptive_thresholded_image.jpg", th3)

    res = cv2.threshold(
        th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # cv2.imwrite("4final_image.jpg", res)

    return res
