import pickle
import cv2
import numpy as np

# destination dimensions of warp
width, height = 1920, 1080

# Points array has Tuples of coordinates -> Global
points = np.zeros((4, 2), int)
counter = 0


def mousepoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN and counter < 4:
        points[counter] = x, y
        counter += 1
        print(f"Point added: {(x, y)}")


def warpImage(image, pts, size=(1920, 1080)):
    original_points = np.array(pts, dtype=np.float32)
    destination_points = np.array([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(original_points, destination_points)
    warpedOutput = cv2.warpPerspective(image, matrix, (size[0], size[1]))
    return warpedOutput, matrix


# Connect to WebCam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

cv2.namedWindow("Original Image")
cv2.setMouseCallback("Original Image", mousepoints)

while True:
    success, img = cap.read()

    if not success or img is None or img.size == 0:
        print("Failed to capture image from webcam. Retrying...")
        continue

    # Draw circles at clicked points on the original image
    for i in range(counter):
        cv2.circle(img, tuple(points[i]), 7, (0, 255, 0), cv2.FILLED)

    # Show original image if points are not selected
    if counter < 4:
        cv2.imshow("Original Image", img)
    else:
        # Warp the image if all four points are selected
        imgOutput, matrix = warpImage(img, points)
        cv2.imshow("Warped Image", imgOutput)

        # Save points to Pickle file after all points are added
        with open("../GetCornerPoints/map.p", "wb") as fileObj:
            pickle.dump(points, fileObj)
            print("Points saved to file: map.p")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('r'):  # Press 'r' to reset points
        counter = 0
        points = np.zeros((4, 2), int)
        print("Points reset")

# Release Resources from Capture
cap.release()
cv2.destroyAllWindows()