import cv2
import numpy as np


class RedButtonDetector:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("[Error] Video not accessible")
            exit()

        cv2.namedWindow("Sliders")
        cv2.namedWindow("Info")
        self.create_sliders("Sliders")
        self.show_motion_status_window("Pending action")
        self.prev_button_areas = []
        self.center_ema = None
        self.alpha = 0.5

    def preprocess_frame(self, frame):
        """
        Convert the input frame to HSV and apply a Gaussian blur to reduce noise.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        return blurred

    def apply_color_threshold(self, frame, lower_color, upper_color, erode, dilate):
        """
        Threshold the input frame in the color range and apply morphological operations to reduce noise and enhance the object.
        """
        mask = cv2.inRange(frame, lower_color, upper_color)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode)
        mask = cv2.dilate(mask, kernel, iterations=dilate)
        return mask

    def find_red_button(self, mask, min_circularity=0.8, min_area=500):
        """
        Find the best contour (red button) based on circularity and size constraints, and return its center.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        center = None
        max_circularity = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            # Update min_circularity based on area
            adaptive_min_circularity = min_circularity * (1 - min(0.5, (area - min_area) / (5 * min_area)))

            if area > min_area and circularity > adaptive_min_circularity and circularity > max_circularity:
                max_circularity = circularity
                best_contour = contour

        if best_contour is not None:
            moments = cv2.moments(best_contour)
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            center = (center_x, center_y)

        return best_contour, center


    def update_center_ema(self, center):
        """
        Update the exponential moving average (EMA) of the red button's center position.
        """
        if self.center_ema is None:
            self.center_ema = center
        elif center is not None:
            self.center_ema = (
                int(self.alpha * center[0] + (1 - self.alpha) * self.center_ema[0]),
                int(self.alpha * center[1] + (1 - self.alpha) * self.center_ema[1])
            )
        return self.center_ema

    def draw_red_button(self, frame, contour, center):
        """
        Draw the detected red button and its center on the input frame.
        """
        if contour is not None and center is not None:
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    def create_sliders(self, window_name):
        """
        Create sliders for color range and morphological operation iterations.
        """
        cv2.createTrackbar("Lower H", window_name, 129, 179, self.on_trackbar)
        cv2.createTrackbar("Upper H", window_name, 179, 179, self.on_trackbar)
        cv2.createTrackbar("Lower S", window_name, 28, 255, self.on_trackbar)
        cv2.createTrackbar("Upper S", window_name, 255, 255, self.on_trackbar)
        cv2.createTrackbar("Lower V", window_name, 33, 255, self.on_trackbar)
        cv2.createTrackbar("Upper V", window_name, 255, 255, self.on_trackbar)

        cv2.createTrackbar("Erode Iterations", window_name, 1, 10, self.on_trackbar)
        cv2.createTrackbar("Dilate Iterations", window_name, 7, 10, self.on_trackbar)

    def on_trackbar(self, *args):
        pass

    def track_button_area(self, contour):
        """
        Track the area of the red button and determine if it is pushed.
        """
        area = cv2.contourArea(contour) if contour is not None else 0
        self.prev_button_areas.append(area)

        # Keep track of the last few button areas (e.g., last 10 frames)
        if len(self.prev_button_areas) > 10:
            self.prev_button_areas.pop(0)

        # Check if the area has been constantly increasing and is larger than a certain threshold
        if (
            len(self.prev_button_areas) == 10
            and all([self.prev_button_areas[i] < self.prev_button_areas[i + 1] for i in range(len(self.prev_button_areas) - 1)])
            and area > 20000  # Set an appropriate threshold for your application
        ):
            return True
        else:
            return False

    def show_motion_status_window(self, text):
        """
        Show the current motion status, i.e. Move forward, rotate, ...
        """
        # Create a black background
        height, width = 480, 640
        background = np.zeros((height, width, 3), np.uint8)

        # Set the text color, font, and size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 2

        # Get the size of the text
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Calculate the position to center the text
        x = (width - text_size[0]) // 2
        y = (height - text_size[1]) // 2

        # Add the text to the background
        cv2.putText(background, text, (x, y), font, font_scale, font_color, font_thickness)

        # Show the info window
        cv2.imshow("Info", background)

    def update_info_text(self, text):
        """
        Updates the motion status text, i.e. Move forward, rotate, ..., in the motion status window
        """
        self.show_motion_status_window(text)

    def decide_motion(self, center, frame):
        """
        Decide the motion of the submarine based on the center position of the red button.
        """
        if center is None:
            return "Searching for red button..."

        height, width, _ = frame.shape
        center_x, center_y = center
        half_width = width // 2
        half_height = height // 2

        x_diff = center_x - half_width
        y_diff = center_y - half_height

        # Set thresholds for moving forward, rotating, etc.
        forward_threshold = 100
        rotate_threshold = 50
        vertical_threshold = 50

        if y_diff > forward_threshold:
            motion = "Move forward"
        elif x_diff > rotate_threshold:
            motion = "Rotate right"
        elif x_diff < -rotate_threshold:
            motion = "Rotate left"
        elif y_diff < -vertical_threshold:
            motion = "Move upwards"
        elif y_diff > vertical_threshold:
            motion = "Move downwards"
        else:
            motion = "Hold camera angle"


        return motion

    def run(self):
        while True:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame = cv2.resize(frame, (640, 480))

                preprocessed_frame = self.preprocess_frame(frame)

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

                color_mask = self.apply_color_threshold(preprocessed_frame, lower_color, upper_color, erode, dilate)

                button_contour, button_center_raw = self.find_red_button(color_mask)

                # Update the center's EMA filter
                button_center_smooth = self.update_center_ema(button_center_raw)
                self.draw_red_button(frame, button_contour, button_center_smooth)


                cv2.imshow("Frame", frame)
                cv2.imshow("Preprocessed", preprocessed_frame)
                cv2.imshow("Mask", color_mask)

                # Track the button area and check if it is pushed
                if self.track_button_area(button_contour):
                    motion = "Disengage"
                else:
                    motion = self.decide_motion(button_center_smooth, frame)

                self.update_info_text(motion)


                if cv2.waitKey(16) & 0xFF == ord('q'):
                    break

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.cap.release()
        cv2.destroyAllWindows()
