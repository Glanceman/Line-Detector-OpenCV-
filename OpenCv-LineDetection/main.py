
from imutils.perspective import four_point_transform
import numpy as np
import cv2
from Scanner import Scanner
from InputLineDetector import InputLineDetector

def ResizeInRatio(img):
    height = img.shape[0]
    width = img.shape[1]

    if(height>960):
        ratio=width/height
        height=960
        width=int(height*ratio)
    print(str(width)+" "+str(height))
    temp=cv2.resize(img,(int(width),int(height)))
    return temp

def ReadImage(filePath):
    img =cv2.imread(filePath)
    img=ResizeInRatio(img)
    return img

def ImageProcess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image to gray scale
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #cv2.imshow("Processing_IMG",thresh)
    #cv2.waitKey(0)
    return thresh

def DetectHorziontalLine(img):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    detect_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts,_ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    result=[]
    img_width=img.shape[1]
    for cnt in cnts: #filter too long lines
        arcLength=cv2.arcLength(cnt,False)
        approx = cv2.approxPolyDP(cnt, 0.05 * arcLength,False)
        if(arcLength<(img_width*0.7)):
            result.append(approx)
       
    
    #print([result])

    #for c in cnts:
    #    cv2.drawContours(temp, [c], -1, (0,255,0), 2)
    return [result];
    


def main():
    print("this is main function");
    img_Path="Input_IMG/sample4.jpg"
    img=cv2.imread(img_Path)

    scanner= Scanner()
    scannedImg=img.copy()
    #scannedImg=scanner.Scan(img)
    scannedImg=ResizeInRatio(scannedImg)

    #editImg=ImageProcess(scannedImg)
    #cnts=DetectHorziontalLine(editImg)

    #for i,cnt in enumerate(cnts):
    #    for j,c in enumerate(cnts[i]):
    #        #c=cnts[i][j]
    #        bot_Left=cnts[i][j][0]
    #        top_left=bot_Left.copy()
    #        top_left[0][1]-=20
 
    #        bot_right=cnts[i][j][len(c)-1]   
    #        top_right=bot_right.copy()
    #        top_right[0][1]-=20

    #        #add top-right
    #        cnts[i][j] = np.concatenate((cnts[i][j],[top_right]),axis=0)
    #        #add top-left
    #        cnts[i][j] = np.concatenate((cnts[i][j],[top_left]),axis=0)
    #        print(" ")
    #        print("Rect: ")
    #        print("top_left: "+str(top_left)+" "+"top_right: "+str(top_right))
    #        print("bot_Left: "+str(bot_Left)+" "+"bot_right "+str(bot_right))

    inputLineDetector = InputLineDetector(scannedImg)
    inputLineDetector.ImageProcess()
    inputLineDetector.DetectHorziontalLine(scannedImg.copy())
    cnts=inputLineDetector.SimplifyAndCreate4Pts()
        


    for c in cnts:
        cv2.drawContours(scannedImg, c, -1, (0,255,0), 2)

    #cv2.imshow("Image",img)
    cv2.imshow("ScannedImage",scannedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows();

if __name__ == "__main__":
    main()
