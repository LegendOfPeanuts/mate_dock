# Red Button Detection

This directory contains the code for detecting the red button in the MATE 2023 task.

## Workflow
1. Define color range
2. Get a mask with red pixels in the color range
3. Remove noise with morphological operations
4. Find contours of red regions
5. Loop over contours
   6. If area > certain size
      7. Calculate the center position using moments
      8. Outline the button and its center in imshow
      9. Report the position and (possibly) approximate distance to motion controller
      
