import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(self.minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box # 相對位置，值介於0~1之間，所以要乘上圖片的寬高
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                # bboxC.xmin: 左上角x座標, bboxC.ymin: 左上角y座標, bboxC.width: 寬, bboxC.height: 高
                bboxs.append([id, bbox, detection.score])
                
                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv2.putText(
                        img,
                        f'{int(detection.score[0]*100)}%',
                        (bbox[0], bbox[1]-20),
                        cv2.FONT_HERSHEY_PLAIN, 
                        2, 
                        (0, 255, 0),
                        2
                    )
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h # 右下角座標
        
        cv2.rectangle(img, bbox, (255, 0, 255), rt) # rt: rectangle thickness
        
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img
        
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        img, bboxs = detector.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('Face Detection', img)
        if cv2.waitKey(1) in [ord('q'), 27]:
            break
    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()