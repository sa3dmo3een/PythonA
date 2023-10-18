import cv2 
import numpy as np


cap = cv2.VideoCapture("1.avi")

count_line_postion = 450

algo= cv2.bgsegm.createBackgroundSubtractorMOG()

min_width_react = 80
min_height_react = 80

run = True
while run:
    ret, frame1 = cap.read()
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5)
    
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25,count_line_postion), (1900,count_line_postion),(255,127,0),3)

    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_react) and (h>= min_height_react)
        if not validate_counter:
            continue

        cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,0,255),2)

    #cv2.imshow('Detector', dilatada)

    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:
        run = False



cv2.destroyAllWindows()
cap.release()