
from imutils.perspective import four_point_transform
import numpy as np
import cv2
from BoxDetector import BoxDetector
from Scanner import Scanner
from InputLineDetector import InputLineDetector
import sys

def ResizeInRatio(img):
    height = img.shape[0]
    width = img.shape[1]

    if(height>1080):
        ratio=width/height
        height=1080
        width=int(height*ratio)
    print(str(width)+" "+str(height))
    temp=cv2.resize(img,(int(width),int(height)))
    return temp

def ReadImage(filePath):
    img =cv2.imread(filePath)
    img=ResizeInRatio(img)
    return img

def ProccessArgv(argvs):
    path=""
    mode=""
    output_Path=""
    for i,argv in enumerate(argvs):
        match i:
            case 1:
                path=argv
                break
            case 2:
                mode=argv
                break
            case 3:
                output_Path=argv
                break
    print(path+" "+mode+" "+output_Path)
    return path, mode ,output_Path
    

def main(argvs):
         
    index=3
    while(index<=3):
        img_Path="Input_IMG/sample"+str(index)+".jpg"
        img=cv2.imread(img_Path)

        scanner= Scanner()
        scannedImg=img.copy()
        #scannedImg=scanner.Scan(img)
        #scannedImg=ResizeInRatio(scannedImg)

        inputLineDetector = InputLineDetector(scannedImg)
        inputLineDetector.ImageProcess()
        inputLineDetector.DetectHorziontalLine(scannedImg.copy())
        cnts=inputLineDetector.Create4Pts()
        
        boxDetector=BoxDetector(scannedImg)
        boxDetector.ImageProcess()
        stats=boxDetector.DetectBox()
        CheckBoxIMG=scannedImg.copy()
        
        for x,y,w,h in stats:
            #if(abs(w-h)<10):
                cv2.rectangle(CheckBoxIMG,(x,y),(x+w,y+h),(255,255,0),2)
        CheckBoxIMG=ResizeInRatio(CheckBoxIMG)
        # for x,y,w,h in rects:
        #      #if(abs(w-h)<10):
        #         cv2.rectangle(CheckBoxIMG,(x,y),(x+w,y+h),(255,0,0),2)

        # for cnt in stats:
        #     cv2.drawContours(CheckBoxIMG,cnt,-1,(255,255,0),2)
        #CheckBoxIMG=ResizeInRatio(CheckBoxIMG)
        cv2.imshow("Box",CheckBoxIMG)
        cv2.waitKey(0)


        for c in cnts:
            cv2.drawContours(scannedImg, c, -1, (0,255,0), 2)
        
        scannedImg=ResizeInRatio(scannedImg)
        #cv2.imshow("Image",img)
        cv2.imshow("ScannedImage"+str(index),scannedImg)
        index=index+1

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    inputLineDetector=None #free memory

if __name__ == "__main__":
    main(sys.argv)
