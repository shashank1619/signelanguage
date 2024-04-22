import cv2
import numpy as np
import os
import string

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")

# Define characters for which directories need to be created
characters = string.digits + string.ascii_uppercase

# Create directories for each character in train and test sets
for char in characters:
    if not os.path.exists(f"data/train/{char}"):
        os.makedirs(f"data/train/{char}")
    if not os.path.exists(f"data/test/{char}"):
        os.makedirs(f"data/test/{char}")

# Train or test
mode = "train"
directory = f"data/{mode}/"
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count = {char: len(os.listdir(directory + char)) for char in characters}

    # Printing the count in each set to the screen
    cv2.putText(
        frame, f"MODE : {mode}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1
    )
    cv2.putText(
        frame, "IMAGE COUNT", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1
    )
    y_offset = 70
    for char, char_count in count.items():
        cv2.putText(
            frame,
            f"{char} : {char_count}",
            (10, y_offset),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 255),
            1,
        )
        y_offset += 10

    cv2.rectangle(frame, (220 - 1, 9), (620 + 1, 419), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[10:410, 220:520]

    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    ret, test_image = cv2.threshold(
        th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    test_image = cv2.resize(test_image, (300, 300))
    cv2.imshow("test", test_image)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break
    # Saving images based on the key pressed
    for i, char in enumerate(characters):
        if interrupt & 0xFF == ord(char):
            cv2.imwrite(f"{directory}{char}/{count[char]}.jpg", roi)
    for i, char in enumerate(string.ascii_lowercase):
        if interrupt & 0xFF == ord(char):
            cv2.imwrite(f"{directory}{char.upper()}/{count[char.upper()]}.jpg", roi)

cap.release()
cv2.destroyAllWindows()
