import cv2
import numpy as np


class RedButtonDetector:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("[Error] Video not accessible")
            exit()

        cv2.namedWindow("Sliders")
        self.create_sliders("Sliders")

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

    def find_red_button(self, mask, min_circularity=0.8, min_area=1000):
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

            if area > min_area and circularity > min_circularity and circularity > max_circularity:
                max_circularity = circularity
                best_contour = contour

        if best_contour is not None:
            moments = cv2.moments(best_contour)
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            center = (center_x, center_y)

        return best_contour, center

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

    def run(self):
        while True:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                if not ret:
                    print("[Warning] Frame skipped")
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

                button_contour, button_center = self.find_red_button(color_mask)

                self.draw_red_button(frame, button_contour, button_center)

                cv2.imshow("Frame", frame)
                cv2.imshow("Preprocessed", preprocessed_frame)
                cv2.imshow("Mask", color_mask)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.cap.release()
        cv2.destroyAllWindows()
