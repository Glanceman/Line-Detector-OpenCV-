
from imutils.perspective import four_point_transform
import numpy as np
import cv2

class InputLineDetector:

    def __init__(self, img):
        self.img = img
        self.binaryImg=None
        self.lineContours=None

    def ImageProcess(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) # convert the image to gray scale
        #self.img = cv2.GaussianBlur(self.img, (3, 3), 0)
        self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        self.binaryImg=self.img.copy()


    def SortPointInContour(self,cnt):
        #print("Sorting")
        #print(cnt)
        #print ("Sorted")
        cnt = sorted(cnt , key=lambda k: k[0][0])
        #print(np.array(cnt))
        return np.array(cnt)
        
        

    def DetectHorziontalLine(self,tempImg):
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        detect_horizontal = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        #cv2.imshow("Temp",detect_horizontal)
        #cv2.waitKey(0)
        cnts,_ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result=[]
        img_width=self.img.shape[1]
        for cnt in cnts: #filter too long lines
            cnt=self.SortPointInContour(cnt)
            arcLength=cv2.arcLength(cnt,False)
            approx = cv2.approxPolyDP(cnt, 0.1 * arcLength,False)
            if(arcLength<(img_width*0.7)):
                result.append(approx)
        print("Result",result)
        self.lineContours=[result]


    def FindWhiteRegionInLine(self,imgRegion,boxSize,direction=1):
        #cv2.imshow("Region: ",imgRegion)
        regionWidth=imgRegion.shape[1]
        if(direction==1):
            index=0
            while(index*boxSize<int(regionWidth*0.4)):
                imgsubRegion=imgRegion[ 0 : boxSize , index*boxSize : index*boxSize+boxSize ]
                if(np.sum(imgsubRegion)==0):
                    return index*boxSize
                index=index+1
        elif(direction==-1):
            index=0
            leftSide = regionWidth-(index*boxSize+boxSize)
            rightSide = regionWidth-index*boxSize
            while(leftSide>int(regionWidth*0.6)): 
                imgsubRegion=imgRegion[ 0 : boxSize , leftSide : rightSide]
                sumOfMatrix=np.sum(imgsubRegion)
                if(sumOfMatrix==0):
                    return index*boxSize
                index=index+1
                leftSide = regionWidth-(index*boxSize+boxSize)
                rightSide = regionWidth-index*boxSize
        return -1

    def SimplifyAndCreate4Pts(self):
        for i,cnt in enumerate(self.lineContours):
            for j,c in enumerate(self.lineContours[i]):
                #c=cnts[i][j]
                origin_Bot_Left=self.lineContours[i][j][0]
                origin_bot_right=self.lineContours[i][j][len(c)-1]
                if(origin_Bot_Left[0][0]==29):
                    print("Dummy")
                #find max predictedHeight check left side first
                predictedHeight_Max=30
                shiftUp=1 #used to move checking box above the line to prevent wrong detection
                boxSize=15 #smaller more accurate but need more performance !find a balance value
                predictedHeight=0
                while(predictedHeight<predictedHeight_Max):#find the first impossible height within the set limited
                    if(predictedHeight!=0):
                        roi=self.binaryImg[ origin_Bot_Left[0][1]-predictedHeight-shiftUp : origin_Bot_Left[0][1]-predictedHeight +boxSize-shiftUp, origin_Bot_Left[0][0] : origin_Bot_Left[0][0]+boxSize ]#y,x small to large
                        if(np.sum(roi)!=0):
                            break
                    predictedHeight+=boxSize

                #find possible place
                #predictedHeight=20 # hardcoded 
                #boxSize=20
                offsetFromStart=0
                while(predictedHeight>0):#todo scan from start to end
                    roi=self.binaryImg[ origin_Bot_Left[0][1]-predictedHeight -shiftUp: origin_Bot_Left[0][1]-predictedHeight+boxSize -shiftUp, origin_Bot_Left[0][0] : origin_bot_right[0][0] ]#y,x small to large
                    offsetFromStart=self.FindWhiteRegionInLine(roi,boxSize)
                    if(offsetFromStart!=-1):
                        break
                    else:
                        predictedHeight=predictedHeight-boxSize
                #scan from end to start
                offsetFromEnd=0
                while(predictedHeight>0):
                    roi=self.binaryImg[ origin_Bot_Left[0][1]-predictedHeight-shiftUp : origin_Bot_Left[0][1]-predictedHeight+boxSize-shiftUp , origin_Bot_Left[0][0] : origin_bot_right[0][0] ]#y,x small to large
                    offsetFromEnd=self.FindWhiteRegionInLine(roi,boxSize,-1)
                    if(offsetFromEnd!=-1):
                        break
                    else:
                        predictedHeight=predictedHeight-boxSize


                #roi=self.binaryImg[bot_Left[0][1]-40:bot_Left[0][1]-20,bot_Left[0][0]:bot_right[0][0]]#y,x small to large
                #print("imgRegion: ")
                #print(roi)
                #offset=self.FindWhiteRegionInLine(roi,20)

                bot_Left=self.lineContours[i][j][0]
                bot_Left[0][0]+=offsetFromStart
                top_left=bot_Left.copy()#to do
                top_left[0][1]-=(predictedHeight+shiftUp)
 
                bot_Right=self.lineContours[i][j][len(c)-1]
                bot_Right[0][0]-=offsetFromEnd
                top_right=bot_Right.copy()
                top_right[0][1]-=(predictedHeight+shiftUp)

                #add top-right
                self.lineContours[i][j] = np.concatenate((self.lineContours[i][j],[top_right]),axis=0)
                #add top-left
                self.lineContours[i][j] = np.concatenate((self.lineContours[i][j],[top_left]),axis=0)
                print(" ")
                print("Rect: ")
                print("top_left: "+str(top_left)+" "+"top_right: "+str(top_right))
                print("bot_Left: "+str(origin_Bot_Left)+" "+"bot_right "+str(origin_bot_right))
                print("Offset ",offsetFromStart," ","OffsetEnd ",offsetFromEnd, "Predicted Height: ", predictedHeight)

        return self.lineContours




