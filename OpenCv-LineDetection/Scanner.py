
from imutils.perspective import four_point_transform
import numpy as np
import cv2

class Scanner:
    def ScanImageProcess(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image to gray scale
        blur = cv2.GaussianBlur(gray, (11, 11), 0) # Add Gaussian blur
        edged = cv2.Canny(blur, 75, 200) # Apply the Canny algorithm to find the edges
        kernel = np.ones((10, 10), np.uint8)
        edged= cv2.dilate(edged,kernel)
        return edged

    def Scan(self,img):
        editImg=img.copy();
        editImg=self.ScanImageProcess(editImg)
        contours, _ = cv2.findContours(editImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            # we approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
            # if we found a countour with 4 points we break the for loop
            # (we can assume that we have found our document)
            if len(approx) == 4:
                doc_cnts = approx
                break

        #cv2.drawContours(img,[doc_cnts], -1, (0,255,0), 3)
        warped = four_point_transform(img, doc_cnts.reshape(4, 2))
        return warped;




