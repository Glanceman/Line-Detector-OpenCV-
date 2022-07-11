import numpy as np
import cv2
import sys
import json
from Scanner import Scanner
from InputLineDetector import InputLineDetector
from BoxDetector import BoxDetector

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

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

def ProccessArgv(argvs):
    path=""
    mode=""
    output_Path=""
    for i,argv in enumerate(argvs):
        if(i==1):
            path=argv
        if(i==2):
            mode=argv
        if(i==3):
            output_Path=argv

    print(" path: "+path+" mode: "+mode+" output: "+output_Path)
    return path, mode ,output_Path
    

def main(argvs):
    path, mode ,output_Path=ProccessArgv(argvs)
    if(path==""):
        print("ERROR, image path can not be empty")
        return
        
    img_Path=path
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
    boxes=boxDetector.DetectBox()
    

    if(mode=="-d"): # save image

        for c in cnts:
            cv2.drawContours(scannedImg, c, -1, (0,255,0), 2)
        
        for x,y,w,h in boxes:
            #if(abs(w-h)<10):
                cv2.rectangle(scannedImg,(x,y),(x+w,y+h),(255,255,0),2)
                
        if(output_Path==""):
            output_Path="./Output"
        cv2.imwrite(output_Path+"/image.jpg",scannedImg)

    #convert to json and write it to file
    table={}
    #datas=[]
    for contour in cnts[0]:
        if(len(contour)==4):
            if "TextBox" not in table:
                table["TextBox"]=[]
            data=contour.flatten().tolist()
            group={}
            group["x"]=data[6]
            group["y"]=data[7]
            group["w"]=data[2]-data[0]
            group["h"]=data[1]-data[7]
            table["TextBox"].append(group)
            #datas.append(data)
    
    if "CheckBox" not in table:
            table["CheckBox"]=[]
    for x,y,w,h in boxes:
        group={}
        group["x"]=int(x)
        group["y"]=int(y)
        group["w"]=int(w)
        group["h"]=int(h)
        table["CheckBox"].append(group)


    with open("Output/data.json", "w", encoding="utf-8") as f:
        f.seek(0) # absolute file positioning
        f.truncate() # to erase all data 
        json.dump(table, f, ensure_ascii=False, indent=4)
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    inputLineDetector=None #free memory

if __name__ == "__main__":
    main(sys.argv)
