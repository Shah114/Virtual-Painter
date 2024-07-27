# Import modules
import cv2
import numpy as np
import os
import time
import HandTrackingModule as htm

# Variables
brushSize = 15
eraserSize = 50
fps = 60
time_per_frame = 1.0 / fps

folderPath = 'Deep Learning/Projects/AI Virtual Painter/Header'
myList = os.listdir(folderPath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)

# Set the first image as the header
header = overlayList[0]

# Color
drawColor = (255, 0, 255)

# Set up the camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Set width
cap.set(4, 720) # Set height

# Assigning Detector
detector = htm.handDetector(detectionCon=0.85)

# Previos points
xp, yp = 0, 0

# Making Image Canvas
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Main part
while True:
    start_time = time.time() # Start time for the frame
    # 1. Import the image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If selection Mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # Checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 720:
                    header = overlayList[1]
                    drawColor = (255, 255, 0)
                elif 780 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserSize)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserSize)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushSize)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushSize)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:125, 0:1280] = header

    # Display the image
    cv2.imshow("Virtual Painter", img)
    cv2.imshow("Canvas", imgCanvas)

    # Calculate the elapsed time and sleep for the remaining time to maintain 60 FPS
    elapsed_time = time.time() - start_time
    if elapsed_time < time_per_frame:
        time.sleep(time_per_frame - elapsed_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()