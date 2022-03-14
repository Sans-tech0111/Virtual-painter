import mediapipe as mp
import cv2 as cv
import numpy as np
import os

folderPath = "painter"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imgPath in myList:
    img = cv.imread(f'{folderPath}/{imgPath}')
    overlayList.append(img)
print(len(overlayList))

header = overlayList[0]
drawColor = (0,0,255)
xp,yp=0,0


cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mpHand = mp.solutions.hands
hands = mpHand.Hands(min_tracking_confidence=0.3)
mpDraw = mp.solutions.drawing_utils

imgCanvas = np.zeros((720,1280,3),np.uint8)


while True:

    #import image
    ss, img = cap.read()
    img = cv.flip(img,1)
    
    # find landmarks
    lmlist = []
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks[0]
        # mpDraw.draw_landmarks(img,lms,mpHand.HAND_CONNECTIONS)
        for id,lm in enumerate(lms.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            lmlist.append([id,cx,cy])

    if(len(lmlist)!=0):

        x1,y1= lmlist[8][1:]
        x2,y2= lmlist[6][1:]
        x3,y3= lmlist[12][1:]
        x4,y4= lmlist[10][1:]

    # check which fingers are up
     
    # Two finger are up
        if(y1<y2 and y3<y4):
            xp, yp = 0, 0
            cv.rectangle(img,(x1,y1-25),(x3,y3+25),drawColor,-1)
            if y1<116:
                if 200<x1<420:
                    header = overlayList[0]
                    drawColor = (0,0,255)

                elif 420<x1<610:
                    header = overlayList[1]
                    drawColor = (255,0,0)

                elif 610<x1<810:
                    header = overlayList[2]
                    drawColor = (0,255,0)

                elif 810<x1<1010:
                    header = overlayList[3]
                    drawColor = (255,255,255)

                elif 1095<x1<1280:
                    header = overlayList[4]
                    drawColor = (0,0,0)

    # Index finger up
        if(y1<y2 and y3>y4):
            cv.circle(img,(x1,y1),10,drawColor,-1)
            if xp == 0 and yp == 0:
                xp,yp = x1,y1

            if drawColor == (0,0,0):
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,50)
                cv.line(img,(xp,yp),(x1,y1),drawColor,50)
            else:
                cv.line(img,(xp,yp),(x1,y1),drawColor,8)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,8)
            xp, yp = x1, y1

    imgGray = cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img,imgCanvas)



    #setting header image
    img[0:116,0:1280] = header
    cv.imshow("Image",img)
    cv.waitKey(1)
