"""
Info : Contour is from left of screen to right
Date : Aug. 3rd 2022
"""
import torch
import DataLoad
import numpy as np
import cv2
COLOR = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
def main():
    step = 0
    OrigImg = cv2.imread(r'License_Plate_Image/2.png')
    imgGray = cv2.cvtColor(OrigImg, cv2.COLOR_BGR2GRAY)
    ret, imgBinary = cv2.threshold(imgGray, 150, 200, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    cv2.imshow('Binary', imgBinary)
    dilation = cv2.dilate(imgBinary, kernel, iterations=1)
    #erosion = cv2.erode(dilation, kernel, iterations=2)
    Image = dilation.copy()
    cv2.imshow('imgBinary', Image)

    contours, hierachy = cv2.findContours(Image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 800 and area <1000:
            #cv2.drawContours(img, contour, -1, (0, 255, 255), 3) 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(OrigImg, (x, y), (x+w, y+h), COLOR[step%3], 3) 
            cv2.putText(OrigImg, f'{step}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR[step%3], 2)
            step+=1
    cv2.imshow('DetectAera', OrigImg)

    

    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()