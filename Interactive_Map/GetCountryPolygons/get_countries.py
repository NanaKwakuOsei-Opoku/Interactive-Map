import cv2
import numpy as np
import pickle
from get_map import warpImage  # Ensure this import is correct according to your directory structure
from cvzone.HandTrackingModule import HandDetector

# Global variables
width, height = 1920, 1080
current_polygon = []
polygons = []
counter = 0
map_file_path = "/Users/nosei-opoku/Desktop/MyProjects/Interactive_Map/GetCornerPoints/map.p"
countries_file_path = "countries.p"

# Load map points
with open(map_file_path, "rb") as fileObj:
    map_points = pickle.load(fileObj)  # 4 corner points loaded from other directory
print(f"Loaded map coordinates: {map_points}")

# Load previously defined Regions of Interest polygons from file, if exists
# Load previously defined Regions of Interest polygons from file, if exists
try:
    with open(countries_file_path, "rb") as fileObj:
        try:
            polygons = pickle.load(fileObj)
            print(f"Pre-loaded {len(polygons)} countries")
        except EOFError:
            print("The countries file is empty or corrupted. Starting fresh.")
            polygons = []
except FileNotFoundError:
    print(f"No pre-loaded countries found. Starting fresh.")
    polygons = []


def mousepoints(event, x, y, flags, params):
    global current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        print("CountryPoint added to current polygon:", (x, y))


# To center the names of the countries within boundaries of the polygon
def calculate_centroid(Polygon):
    polygon_array = np.array(Polygon, dtype=np.float32)
    length = polygon_array.shape[0]
    sum_x = np.sum(polygon_array[:, 0])  # for all rows select 1st column elements
    sum_y = np.sum(polygon_array[:, 1])  # for all rows select 2nd column elements
    return int(sum_x / length), int(sum_y / length)


# Connect to WebCam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Processing....
while True:
    success, img = cap.read()

    # Warp the image using the loaded map points
    imgWarped, matrix = warpImage(img, map_points)

    key = cv2.waitKey(1)

    # Save polygons with country name
    if key == ord("s") and len(current_polygon) > 2:
        country_name = input("Enter the country name: ")
        polygons.append([current_polygon, country_name])
        current_polygon = []  # reset for next country
        counter += 1
        print(f"Number of countries saved: {len(polygons)}")

    # Save polygons to countries file
    if key == ord("w"):
        if polygons:  # Only save if there are polygons to save
            with open(countries_file_path, "wb") as fileObj:
                pickle.dump(polygons, fileObj)
                print(f"Saved {len(polygons)} countries to file")
        else:
            print("No countries to save. Add some countries before saving.")

    # To delete recently added polygon if there is mistake
    if key == ord("d") and polygons:
        polygons.pop()

    # Draw the current polygon
    if current_polygon:
        cv2.polylines(imgWarped, [np.array(current_polygon, np.int32)], isClosed=False, color=(255, 0, 0), thickness=5)

    overlay = imgWarped.copy()

    # Draw collected polygons on image
    for polygon, name in polygons:
        cv2.polylines(imgWarped, [np.array(polygon, np.int32)], isClosed=True, color=(255, 0, 0), thickness=5)
        cv2.fillPoly(overlay, [np.array(polygon, np.int32)], color=(255, 0, 0))
        centroid = calculate_centroid(polygon)
        cv2.putText(imgWarped, name, centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, 0.5, imgWarped, 0.5, 0, imgWarped)

    # Set the mouse callback to capture points on the warped image
    cv2.setMouseCallback("Warped Image", mousepoints)

    # Detect hands and get fingertip position
    hands, img = detector.findHands(img, flipType=False)
    if hands:
        first_hand = hands[0]
        normal_finger_tip = first_hand["lmList"][8][0:2]
        cv2.circle(img, tuple(normal_finger_tip), 10, (0, 0, 255), cv2.FILLED)

        warped_finger_tip = cv2.perspectiveTransform(np.array([[[normal_finger_tip[0], normal_finger_tip[1]]]], dtype=np.float32), matrix)[0][0]
        warped_finger_tip = (int(warped_finger_tip[0]), int(warped_finger_tip[1]))
        cv2.circle(imgWarped, warped_finger_tip, 10, (0, 255, 0), cv2.FILLED)

        # Check if fingertip is within any polygon
        for polygon, name in polygons:
            if cv2.pointPolygonTest(np.array(polygon, np.int32), warped_finger_tip, False) >= 0:
                print(f"Finger is within the polygon: {name}")

    cv2.imshow("Warped Image", imgWarped)

    if key & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
