import numpy as np
import cv2

class BoxDetector:
    def __init__(self, img):
        self.img = img
        self.imgWidth=img.shape[1]
        self.imgHeight=img.shape[0]
        self.binaryImg=None
        self.processedIMG=None
        self.lineContours=None
        self.kernlRectLen=int(self.imgWidth*0.02)
    
    def ImageProcess(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) # convert the image to gray scale
        # kernel = np.ones((3,3), np.uint8)
        # self.img=cv2.erode(self.img,kernel,iterations=1) 
        self.binaryImg=cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        line_min_width = int(self.imgWidth*0.012)
        kernal_h = np.ones((1,line_min_width), np.uint8)
        kernal_v = np.ones((line_min_width,1), np.uint8)
        img_bin_h = cv2.morphologyEx(self.binaryImg, cv2.MORPH_OPEN, kernal_h)
        img_bin_v = cv2.morphologyEx(self.binaryImg, cv2.MORPH_OPEN, kernal_v)
        img_bin_final=img_bin_h|img_bin_v
        final_kernel = np.ones((3,3), np.uint8)
        img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=1)   
        self.processedIMG=img_bin_final.copy()


    def DetectBox(self):
        _, labels, stats,_=cv2.connectedComponentsWithStats(self.processedIMG, connectivity=8, ltype=cv2.CV_32S)
        boxes=[]
        for x,y,w,h,area in stats[2:]:
            if(abs(1-(w/h))>0.15):
                continue
            offsetX=int(w*0.25)
            offsetY=int(h*0.25)
            roi=self.binaryImg[ y+offsetY : y+h-offsetY, x+offsetX : x+w-offsetX]#y,x small to large
            # cv2.imshow("roi",roi)
            # cv2.waitKey(0)
            if(np.sum(roi)!=0):
                continue
            boxes.append([x,y,w,h])

        print("Boxs Arr", boxes)
        return boxes
    
    #other approach
    def DetectBox0(self):
        cnts,_ = cv2.findContours(self.processedIMG, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        rects=[]
        for cnt in cnts:
            arcLength=cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.1 * arcLength,True)
            if(len(approx)!=4):
                continue

            if(cv2.contourArea(approx)>self.imgHeight*self.imgHeight*0.5):
                continue

            (x, y, w, h) = cv2.boundingRect(approx)
            if(abs(1-(w/h))>0.15):
                continue

            print("approx", approx)
            boxes.append(cnt)
            rects.append([x,y,w,h])
        return boxes,rects

