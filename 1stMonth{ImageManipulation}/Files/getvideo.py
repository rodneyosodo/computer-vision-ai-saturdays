import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 60.0, (640,480))
# 1.Name
# 2.fourcc 4-character code of codec used to compress the frames. 
# For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, 
# VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc.
# 3.fps
# 4.frameSize	Size of the video frames

while(True):
    #  ret is a boolean regarding whether or not there was a return at all,
    #  at the frame is each frame that is returned. If there is no frame, you wont get an error, you will get None.
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
