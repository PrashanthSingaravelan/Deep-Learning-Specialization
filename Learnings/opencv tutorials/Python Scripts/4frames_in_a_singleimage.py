import numpy as np
import cv2

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    width  = int(cap.get(3)) ## returns the float, so convert into int
    height = int(cap.get(4))
    
    image         = np.zeros(frame.shape, np.uint8)  ## an empty list with the captured frame rate
    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

    ## top-left
    image[:height//2, :width//2] = smaller_frame
    
    ## bottom-left
    image[height//2:, :width//2] = smaller_frame
    
    ## top-right
    image[:height//2, width//2:] = smaller_frame
    
    ## bottom-right
    image[height//2:, width//2:] = smaller_frame

    cv2.imshow('frame', image)
    
    
    ## cv2.waitKey(1) returns the ASCII character of the key pressed
    if cv2.waitKey(1) == ord('q'):  ## So if 'q' is pressed, then the video capture breaks
        break

cap.release() ## releasing the hold camera resource
cv2.destroyAllWindows()