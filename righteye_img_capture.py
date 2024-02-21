import cv2
import time
import numpy as np
import detectors.facemesh as fm
import pyautogui
import copy
from utils import capture_eye_region, save_eye_image, get_relative_mouse_position, create_eye_region_mask, resize_and_pad_image


def initialize_camera(width=640, height=360):
    cap = cv2.VideoCapture("Video.mp4")
    cap.set(3, width)
    cap.set(4, height)
    return cap

def find_center(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    center_x = int(sum(x) / len(points))
    center_y = int(sum(y) / len(points))
    return center_x, center_y

def draw_cross(frame, center, size=5, color=(255, 255, 255), thickness=2):
    cv2.line(frame, (center[0] - size, center[1]), (center[0] + size, center[1]), color, thickness)
    cv2.line(frame, (center[0], center[1] - size), (center[0], center[1] + size), color, thickness)
    
    
def map_eye_position_to_screen(center_x, center_y, x_min, x_max, y_min, y_max, wScr, hScr):
    screen_x = np.interp(center_x, [x_min, x_max], [0, wScr])
    screen_y = np.interp(center_y, [y_min, y_max], [0, hScr])
    return screen_x, screen_y

def smoothen_movement(plocX, plocY, x3, y3, smoothening=7):
    clocX = plocX + (x3 - plocX) / smoothening
    clocY = plocY + (y3 - plocY) / smoothening
    return clocX, clocY



def main():
    wCam, hCam = 640, 360
    cap = initialize_camera(wCam, hCam)
    detector = fm.FaceMeshDetector(maxFaces=1, refine_landmarks=True)
    pTime = 0
    idList = [474, 475, 476, 477]
    bboxList = [336, 293, 261, 276]
    image_counter = 0
    capturing = False
    while True:
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == cap.get(cv2.CAP_PROP_POS_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
        success, frame_ori = cap.read()
        if not success:
            print("can't read frame.")
            break
        frame = copy.deepcopy(frame_ori)
        frame = cv2.flip(frame, 1)
        frame, faces = detector.findFaceMesh(frame, draw=False)
        if faces:
            face = faces[0]
            eye_points = [face[id] for id in idList]
            bbox_points = [face[id] for id in bboxList]
            #center = find_center(eye_points)
            #draw_cross(frame, center)
            x_min, x_max = min(point[0] for point in bbox_points), max(point[0] for point in bbox_points)
            y_min, y_max = min(point[1] for point in bbox_points), max(point[1] for point in bbox_points)
            
            if cv2.waitKey(1) & 0xFF == 13: # Enter
                capturing = not capturing
                print(f"Capturing: {capturing}")
            eye_region_mask = create_eye_region_mask(frame, x_min, x_max, y_min, y_max)
            resize_and_padded_eye_region = resize_and_pad_image(eye_region_mask, target_size=(448, 448))     
            if capturing:
                print("Capturing eye region...")
                #eye_region = capture_eye_region(frame, x_min, x_max, y_min, y_max)
                relative_mouse_x, relative_mouse_y = get_relative_mouse_position()
                file_name = f"x{relative_mouse_x:.3f}_y{relative_mouse_y:.3f}.jpg"
                save_eye_image(resize_and_padded_eye_region, file_name=file_name)
                image_counter += 1
                
            for x, y in eye_points:
                cv2.circle(frame, (x, y), 3, (255, 0, 255), cv2.FILLED)
            cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), (0, 255, 0), 2)
                
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)        
        cv2.imshow('image', frame)
        cv2.imshow('eye_region', eye_region_mask)
        if cv2.waitKey(1) in [ord('q'), 27]:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()