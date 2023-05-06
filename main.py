from vision.vision import RedButtonDetector


# The main running process.

def main():
    video_path = "vision/test_data/videos/surface_to_dock_trimmed.mp4"  # Testing video path
    red_button_detector = RedButtonDetector(video_path)  # This class is in the root/vision/vision.py script
    red_button_detector.run()


if __name__ == "__main__":
    main()
