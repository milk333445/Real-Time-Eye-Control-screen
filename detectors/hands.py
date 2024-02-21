import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, model_complexity=1, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
                static_image_mode=self.mode,
                max_num_hands=self.maxHands,
                min_detection_confidence=self.detectionCon,
                min_tracking_confidence=self.trackCon,
                model_complexity=self.model_complexity,
                
            )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIDs = [4, 8, 12, 16, 20] # 4:大拇指指尖, 8:食指指尖, 12:中指指尖, 16:無名指指尖, 20:小指指尖
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for idx, handLms in enumerate(self.results.multi_hand_landmarks):
                handType = "Right" if self.results.multi_handedness[idx].classification[0].label == "Left" else "Left"
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    #self.drawHandType(img, handLms, handType)
        return img
    
    def drawHandType(self, img, handLms, handType):
        xList = [landmark.x for landmark in handLms.landmark]
        yList = [landmark.y for landmark in handLms.landmark]
        xmin, xmax = int(min(xList) * img.shape[1]), int(max(xList) * img.shape[1])
        ymin, ymax = int(min(yList) * img.shape[0]), int(max(yList) * img.shape[0])
        cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)
        cv2.putText(img, f"{handType}", (xmin-10, ymin-30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] # 0:第一隻手, 1:第二隻手
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax # 代表手部的範圍, 如果有兩隻手，則會有兩個bbox
            
            if draw:
                cv2.rectangle(
                    img,
                    (bbox[0] - 20, bbox[1] - 20),
                    (bbox[2] + 20, bbox[3] + 20),
                    (0, 255, 0),
                )
            
        return self.lmList, bbox
    
    def fingersUp(self):
        fingers = []
        
        if len(self.lmList) == 0:
            print("lmList is empty.")
            return fingers
        
        # thumb
        try:
            thumb_tip_x = self.lmList[self.tipIDs[0]][1]
            thumb_lower_joint_x = self.lmList[self.tipIDs[0] - 1][1]
            fingers.append(1 if thumb_tip_x > thumb_lower_joint_x else 0)
        except IndexError:
            print(f"Error: Index out of range for thumb in lmList.")
            return fingers
        # 4 Fingers
        for id in range(1, 5):
            try:
                finger_tip_y = self.lmList[self.tipIDs[id]][2]
                finger_lower_joint_y = self.lmList[self.tipIDs[id] - 2][2]
                fingers.append(1 if finger_tip_y < finger_lower_joint_y else 0)
            except IndexError:
                print(f"Error: Index out of range for finger {id} in lmList.")
                return fingers
        return fingers
    
    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1) # 計算兩點之間的距離
        return length, img, [x1, y1, x2, y2, cx, cy]
        
            
            
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        img = detector.findHands(img)
        lmList0, bbox0 = detector.findPosition(img, handNo=0, draw=True)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("test Image", img)
        if cv2.waitKey(1) in [ord('q'), 27]:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
            