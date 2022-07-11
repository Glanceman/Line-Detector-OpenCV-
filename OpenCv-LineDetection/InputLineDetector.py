from imutils.perspective import four_point_transform
import numpy as np
import cv2

class InputLineDetector:

    def __init__(self, img):
        self.img = img
        self.imgWidth=img.shape[1]
        self.imgHeight=img.shape[0]
        self.binaryImg=None
        self.lineContours=None
        self.kernlRectLen=int(self.imgWidth*0.02)

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
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernlRectLen,1))
        detect_horizontal = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        #cv2.imshow("Temp",detect_horizontal)
        #cv2.waitKey(0)
        cnts,_ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result=[]
        img_width=self.img.shape[1]
        for cnt in cnts: #filter too long lines
            cnt=self.SortPointInContour(cnt) #sorted to prevent line overlapping e.g list: 1->3->2->4 to 1->2->3->4
            arcLength=cv2.arcLength(cnt,False)
            approx = cv2.approxPolyDP(cnt, 0.1 * arcLength,False)
            if(arcLength<(self.imgWidth*0.8)):
                result.append(approx)
        print("Result",[result])
        self.lineContours=[result]


    def FindWhiteRegionInLine(self,imgRegion,boxSize,direction=1,leftPivot=0,rightPivot=0):
        regionWidth=imgRegion.shape[1]
        boxHeight=boxSize
        boxWidth=int(regionWidth/5)
        minAccpetableWidth=int(regionWidth*0.4)
        if(direction==1):
            index=0
            leftPivot=index
            rightSidePosition=boxWidth+index
            while((rightSidePosition<rightPivot) and (rightPivot-leftPivot)>minAccpetableWidth):
                imgSubRegion=imgRegion[ 0 : boxHeight , index : rightSidePosition ]
                sumOfMatrix=np.sum(imgSubRegion)
                if(sumOfMatrix==0):
                    return index
                index=index+1
                rightSidePosition=boxWidth+index
                leftPivot=index
        elif(direction==-1):
            index=0
            leftSidePosition=regionWidth-(boxWidth+index)
            rightSidePosition= regionWidth-index
            rightPivot=rightSidePosition
            while((leftSidePosition>leftPivot) and (rightPivot-leftPivot)>minAccpetableWidth): 
                imgSubRegion=imgRegion[ 0 : boxHeight , leftSidePosition : rightSidePosition]
                sumOfMatrix=np.sum(imgSubRegion)
                if(sumOfMatrix==0):
                    return index
                index=index+1
                leftSidePosition=regionWidth-(boxWidth+index)
                rightSidePosition= regionWidth-index
                rightPivot=rightSidePosition
        return -1

    def Create4Pts(self):
        for i,cnt in enumerate(self.lineContours):
            for j,c in enumerate(self.lineContours[i]):
                origin_Bot_Left_Point=self.lineContours[i][j][0]
                origin_Bot_Right_Point=self.lineContours[i][j][len(c)-1]
                # if(origin_Bot_Left_Point[0][1]==2134):
                #     print("Dummy")
                #find max predictedHeight check left side first
                predictedHeight_Max=int(self.imgHeight*0.03)
                shiftUp=3 #used to move checking box above the line to prevent wrong detection
                trimSize=5
                boxSize=int(predictedHeight_Max*0.2) 
                predictedHeight=0
                offsetFromStartTable=np.array([])
                offsetFromEndTable=np.array([])

                exploreHeight=boxSize
                while( exploreHeight <= predictedHeight_Max):#find the first impossible height within the set limited
                    roi=self.binaryImg[ origin_Bot_Left_Point[0][1]-exploreHeight-shiftUp : origin_Bot_Left_Point[0][1]-exploreHeight+boxSize-shiftUp, origin_Bot_Left_Point[0][0] : origin_Bot_Right_Point[0][0]]#y,x small to large
                    offsetFromStart=self.FindWhiteRegionInLine(roi,boxSize,1,0,roi.shape[1])
                    offsetFromEnd=self.FindWhiteRegionInLine(roi,boxSize,-1,offsetFromStart)
                    if(offsetFromStart==-1 or offsetFromEnd==-1):
                        break
                    else:
                        offsetFromStartTable=np.append(offsetFromStartTable , offsetFromStart)
                        offsetFromEndTable=np.append(offsetFromEndTable , offsetFromEnd)
                        predictedHeight=exploreHeight

                    exploreHeight+=boxSize

                #finalize the top with new temporary left and right
                offsetFromStart_Max=offsetFromStartTable.max() if offsetFromStartTable.size>0 else 0
                offsetFromEnd_Max=offsetFromEndTable.max() if offsetFromEndTable.size>0 else 0
                
                bot_Left_Point_Temp=self.lineContours[i][j][0].copy()
                bot_Left_Point_Temp[0][0]+=offsetFromStart_Max

                bot_Right_Point_Temp=self.lineContours[i][j][len(c)-1].copy()
                bot_Right_Point_Temp[0][0]-=offsetFromEnd_Max
                
                finalizedHeight=boxSize
                while(finalizedHeight<=predictedHeight):
                    new_roi=self.binaryImg[ origin_Bot_Left_Point[0][1]-finalizedHeight-shiftUp : origin_Bot_Left_Point[0][1]-finalizedHeight+boxSize-shiftUp, bot_Left_Point_Temp[0][0] : bot_Right_Point_Temp[0][0]]#y,x small to large
                    sumOfMatrix=np.sum(new_roi)
                    # if(origin_Bot_Left_Point[0][0]==1419 and origin_Bot_Left_Point[0][1]==3156):
                    #     cv2.imshow("ROI",new_roi)
                    #     waitKey(0)
                    if(sumOfMatrix!=0):
                        currentIndex=int(finalizedHeight/boxSize-1)
                        lastIndex=int(len(offsetFromStartTable))
                        offsetFromStartTable=np.delete(offsetFromStartTable,np.s_[currentIndex:lastIndex],None)
                        offsetFromEndTable=np.delete(offsetFromEndTable,np.s_[currentIndex:lastIndex],None)
                        finalizedHeight-=boxSize
                        break
                    finalizedHeight+=boxSize

                if(finalizedHeight>predictedHeight):
                    finalizedHeight=predictedHeight
                
                offsetFromStart_Max=offsetFromStartTable.max() if offsetFromStartTable.size > 0 else 0
                offsetFromEnd_Max=offsetFromEndTable.max() if offsetFromEndTable.size > 0 else 0

                bot_Left_Point=self.lineContours[i][j][0]
                bot_Left_Point[0][0]+=(offsetFromStart_Max+trimSize) #X
                bot_Left_Point[0][1]-=shiftUp #Y
                top_left_Point=bot_Left_Point.copy()
                top_left_Point[0][1]-=finalizedHeight

                bot_Right_Point=self.lineContours[i][j][len(c)-1]
                bot_Right_Point[0][0]-=(offsetFromEnd_Max+trimSize)
                bot_Right_Point[0][1]-=shiftUp
                top_right_Point=bot_Right_Point.copy()
                top_right_Point[0][1]-=finalizedHeight

                if(finalizedHeight>0):
                    #add top-right
                    self.lineContours[i][j] = np.concatenate((self.lineContours[i][j],[top_right_Point]),axis=0)
                    #add top-left
                    self.lineContours[i][j] = np.concatenate((self.lineContours[i][j],[top_left_Point]),axis=0)

                print(" ")
                print("Rect: ")
                print("top_left: "+str(top_left_Point)+" "+"top_right: "+str(top_right_Point))
                print("bot_Left: "+str(origin_Bot_Left_Point)+" "+"bot_right "+str(origin_Bot_Right_Point))
                print("OffsetStart ",offsetFromStart_Max," ","OffsetEnd ",offsetFromEnd_Max, "Predicted Height: ", finalizedHeight)
                print("Size: ",len(self.lineContours[i][j]))


        return self.lineContours




