import cv2
import numpy as np

# Function to preprocess the frame
def preprocess_frame(frame):

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

    return blurred

# Function to apply color thresholding
def apply_color_threshold(frame, lower_color, upper_color, erode, dilate):
    # Threshold the frame in the color range
    mask = cv2.inRange(frame, lower_color, upper_color)

    # Apply morphological operations to reduce noise and enhance the object
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=erode)
    mask = cv2.dilate(mask, kernel, iterations=dilate)

    return mask

# Function to find the red button
def find_red_button(mask, min_circularity=0.8, min_area=1000):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the best contour and its center
    best_contour = None
    center = None

    # Find the best contour based on circularity and size constraints
    max_circularity = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        if area > min_area and circularity > min_circularity and circularity > max_circularity:
            max_circularity = circularity
            best_contour = contour

    if best_contour is not None:
        # Calculate the center of the best contour
        moments = cv2.moments(best_contour)
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        center = (center_x, center_y)

    return best_contour, center

# Function to draw the detected red button and its center
def draw_red_button(frame, contour, center):
    if contour is not None and center is not None:
        # Draw the contour
        cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)

        # Draw the center
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

def create_sliders(window_name):
    # Create sliders for color range
    cv2.createTrackbar("Lower H", window_name, 129, 179, on_trackbar)
    cv2.createTrackbar("Upper H", window_name, 179, 179, on_trackbar)
    cv2.createTrackbar("Lower S", window_name, 28, 255, on_trackbar)
    cv2.createTrackbar("Upper S", window_name, 255, 255, on_trackbar)
    cv2.createTrackbar("Lower V", window_name, 33, 255, on_trackbar)
    cv2.createTrackbar("Upper V", window_name, 255, 255, on_trackbar)


    cv2.createTrackbar("Erode Iterations", window_name, 1, 10, on_trackbar)
    cv2.createTrackbar("Dilate Iterations", window_name, 7, 10, on_trackbar)

def on_trackbar(*args):
    pass

video_path = "videos/red_button_closing_in.MP4"
# Initialize the video capture
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[Error] Webcam not accessible")
    exit()

# Create a window for sliders
cv2.namedWindow("Sliders")

# Create sliders
create_sliders("Sliders")

while True:
    while cap.isOpened():
        # Read a frame

        #path = "red-round-glossy-button-isolated.png"
        #path = "biofoul.png"
        #path = "seagrass.png"


        #frame = cv2.imread(path)
        ret, frame = cap.read()

        # Check if the frame is empty
        if not ret:
            print("[Warning] Frame skipped")
            continue

        frame = cv2.resize(frame, (640, 480))

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Get the color range from sliders
        lower_h = cv2.getTrackbarPos("Lower H", "Sliders")
        upper_h = cv2.getTrackbarPos("Upper H", "Sliders")
        lower_s = cv2.getTrackbarPos("Lower S", "Sliders")
        upper_s = cv2.getTrackbarPos("Upper S", "Sliders")
        lower_v = cv2.getTrackbarPos("Lower V", "Sliders")
        upper_v = cv2.getTrackbarPos("Upper V", "Sliders")
        lower_color = np.array([lower_h, lower_s, lower_v])
        upper_color = np.array([upper_h, upper_s, upper_v])
        erode = cv2.getTrackbarPos("Erode Iterations", "Sliders")
        dilate = cv2.getTrackbarPos("Dilate Iterations", "Sliders")

        # Apply color thresholding
        color_mask = apply_color_threshold(preprocessed_frame, lower_color, upper_color, erode, dilate)

        # Find the red button and its center
        button_contour, button_center = find_red_button(color_mask)

        # Draw the detected red button and its center
        draw_red_button(frame, button_contour, button_center)

        # Show the processed frame and the mask
        cv2.imshow("Frame", frame)
        cv2.imshow("Preprocessed", preprocessed_frame)
        cv2.imshow("Mask", color_mask)

        # Check if the user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
