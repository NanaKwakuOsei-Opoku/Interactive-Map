import cv2
import numpy as np
import pickle
from cvzone.HandTrackingModule import HandDetector
from flight_info_backend import get_flight_info
from GetCountryPolygons.get_map import warpImage

# Global variables
WIDTH, HEIGHT = 1920, 1080
MAP_FILE_PATH = "map.p"
COUNTRIES_FILE_PATH = "countries.p"

# Load map points and country polygons
with open(MAP_FILE_PATH, "rb") as file:
    map_points = pickle.load(file)

with open(COUNTRIES_FILE_PATH, "rb") as file:
    polygons = pickle.load(file)


def inverse_warp_image(img_overlay, pts, image):
    original_points = np.float32(pts)
    destination_points = np.float32([[0, 0], [img_overlay.shape[1] - 1, 0],
                                     [0, img_overlay.shape[0] - 1],
                                     [img_overlay.shape[1] - 1, img_overlay.shape[0] - 1]])
    transformation = cv2.getPerspectiveTransform(destination_points, original_points)
    warped_overlay = cv2.warpPerspective(img_overlay, transformation, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1, warped_overlay, 0.65, 0)


def warp_single_point(point, matrix):
    homogenous_point = np.array([*point, 1], dtype=np.float32)
    homogenous_point_transformed = np.dot(matrix, homogenous_point)
    return homogenous_point_transformed[:2] / homogenous_point_transformed[2]


def get_finger_locations(image, img_warped, matrix, detector):
    hands, _ = detector.findHands(image, draw=False, flipType=True)
    warped_tips = []
    for hand in hands[:2]:  # Process up to two hands
        finger_tip = hand["lmList"][8][:2]
        cv2.circle(image, finger_tip, 5, (0, 0, 255), cv2.FILLED)
        warped_tip = tuple(map(int, warp_single_point(finger_tip, matrix)))
        cv2.circle(img_warped, warped_tip, 7, (0, 255, 0), cv2.FILLED)
        warped_tips.append(warped_tip)
    return warped_tips


def create_overlay_image(polygons, warped_tips, img_overlay):
    detected_countries = []
    for warped_tip in warped_tips:
        for polygon, name in polygons:
            if cv2.pointPolygonTest(np.array(polygon, np.int32), warped_tip, False) >= 0:
                cv2.fillPoly(img_overlay, [np.array(polygon)], color=(255, 0, 0))
                detected_countries.append(name)
    if len(warped_tips) == 2:
        cv2.line(img_overlay, warped_tips[0], warped_tips[1], (0, 255, 255), 5)
    return img_overlay, detected_countries


def add_info_box(image, country1, country2):
    flight_info = get_flight_info(country1, country2)
    if not flight_info:
        return image

    box_width, box_height = 400, 200
    x, y = 50, image.shape[0] - box_height - 50

    cv2.rectangle(image, (x, y), (x + box_width, y + box_height), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(image, (x, y), (x + box_width, y + box_height), (255, 255, 255), 2)

    info_text = [
        ("Flight Information", 1, 40),
        (f"From: {flight_info['from']}", 0.6, 80),
        (f"To: {flight_info['to']}", 0.6, 110),
        (f"Flight Time: {flight_info['flight_time_hours']:.2f} hours", 0.6, 140),
        (f"Distance: {flight_info['distance_km']:.2f} km", 0.6, 170)
    ]

    for text, scale, y_offset in info_text:
        cv2.putText(image, text, (x + 20, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)

    return image


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        img_warped, matrix = warpImage(img, map_points)
        img_overlay = np.zeros_like(img_warped)
        info_box_overlay = np.zeros_like(img)

        finger_locations = get_finger_locations(img, img_warped, matrix, detector)

        if finger_locations:
            img_overlay, countries = create_overlay_image(polygons, finger_locations, img_overlay)
            img_output = inverse_warp_image(img_overlay, map_points, img)

            if len(countries) == 2:
                info_box_overlay = add_info_box(info_box_overlay, countries[0], countries[1])

            img_output = cv2.addWeighted(img_output, 1, info_box_overlay, 1, 0)
        else:
            img_output = img

        cv2.imshow("Interactive Map", img_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
