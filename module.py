import cv2
import mediapipe as mp
import time


class handDetector():

    # Initalise detector as object
    def __init__(self, mode=False, maxHands=2, modelComplex=1,detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils


    # Find the hand and draw
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


    # Get the specific positions of landmarks and add it to a list
    def findPosition(self, img, handNum=0, draw=True):

        landmarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id,lm in enumerate(myHand.landmark):
                 # print(id,lm)
                 h,w,c = img.shape
                 cx,cy = int(lm.x*w), int(lm.y*h)
                 # print(id, cx,cy)
                 landmarkList.append([id, cx, cy])
                 if draw:
                     if id == 0:
                         size = 25
                         colour = (0,255,0)
                     else:
                         size = 10
                         colour = (255,0,0)
                     cv2.circle(img, (cx,cy), size, colour, cv2.FILLED)

        return landmarkList



def main():
    cTime = 0
    pTime = 0

    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success,img = cap.read()

        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        if len(landmarkList) != 0:
            print(landmarkList[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # (img, fps, pos, font, size, colour, thickness)
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()
