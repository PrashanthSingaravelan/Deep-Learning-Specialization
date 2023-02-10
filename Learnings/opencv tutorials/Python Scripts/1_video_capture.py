import numpy as np
import cv2
cap = cv2.VideoCapture(0)  

## frame -> image it-self(numpy array format)
## ret --> if the capture works properly/not\
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    
    ## cv2.waitKey(1) returns the ASCII character of the key pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()