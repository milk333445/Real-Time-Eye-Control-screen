import cv2
import mediapipe as mp
import time
import math

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5, refine_landmarks=False):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.refine_landmarks = refine_landmarks
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode = self.staticMode,
            max_num_faces = self.maxFaces,
            min_detection_confidence = self.minDetectionCon,
            min_tracking_confidence = self.minTrackCon,
            refine_landmarks = self.refine_landmarks
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        #self.mpDraw_styles = mp.solutions.drawing_styles
        
        
    
    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image=img,
                        landmark_list=faceLms,
                        connections=self.mpFaceMesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.drawSpec,
                        connection_drawing_spec=self.drawSpec
                        )
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x, y])
                    
                faces.append(face)
        
        return img, faces

    def findDistance(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info
    
    
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        img, face = detector.findFaceMesh(img)
        #print("len(face):", len(face[0]))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('MediaPipe Face Mesh', img)
        if cv2.waitKey(1) in [ord('q'), 27]:
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()