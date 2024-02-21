import cv2
import time
import numpy as np
import detectors.facemesh as fm
import pyautogui
import copy
from utils import capture_eye_region, save_eye_image, get_relative_mouse_position, create_eye_region_mask, resize_and_pad_image, SmoothMouseMover
from validation import Predictor
import argparse

def initialize_camera(source, width=640, height=360):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
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

def process_frame(frame, idList, bboxList, detector, predictor_x, predictor_y, wScr, hScr, mouse_mover):
    frame, faces = detector.findFaceMesh(frame, draw=False)
    if faces:
        face = faces[0]
        eye_points = [face[id] for id in idList]
        bbox_points = [face[id] for id in bboxList]
        x_min, x_max = min(point[0] for point in bbox_points), max(point[0] for point in bbox_points)
        y_min, y_max = min(point[1] for point in bbox_points), max(point[1] for point in bbox_points)
        
        eye_region_mask = create_eye_region_mask(frame, x_min, x_max, y_min, y_max)
        resize_and_padded_eye_region = resize_and_pad_image(eye_region_mask, target_size=(448, 448))   
        # predict
        pred_x = predictor_x.predict(resize_and_padded_eye_region)
        pred_y = predictor_y.predict(resize_and_padded_eye_region)
        screen_x = pred_x * wScr
        screen_y = pred_y * hScr
        smooth_x, smooth_y = mouse_mover.update_position(screen_x, screen_y)
        pyautogui.moveTo(smooth_x, smooth_y, _pause=False)

        for x, y in eye_points:
            cv2.circle(frame, (x, y), 3, (255, 0, 255), cv2.FILLED)
        cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), (0, 255, 0), 2)
    return frame
def main(args):
    wCam, hCam = args.width, args.height
    cap = initialize_camera(args.source, wCam, hCam)
    detector = fm.FaceMeshDetector(maxFaces=1, refine_landmarks=True)
    pTime = 0
    idList = [474, 475, 476, 477]
    bboxList = [336, 293, 261, 276]
    wScr, hScr = pyautogui.size()
    
    # predictor
    predictor_x = Predictor(args.model_x)
    predictor_y = Predictor(args.model_y)
    
    mouse_mover = SmoothMouseMover(smoothing_factor=args.smoothing_factor)
    
    while True:
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == cap.get(cv2.CAP_PROP_POS_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 設定影片的位置到開頭
        success, frame_ori = cap.read()
        if not success:
            print("can't read frame.")
            break
        frame_ori = cv2.flip(frame_ori, 1)
        frame = copy.deepcopy(frame_ori)
        
        frame = process_frame(frame, idList, bboxList, detector, predictor_x, predictor_y, wScr, hScr, mouse_mover)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)        
        cv2.imshow('image', frame)
        if cv2.waitKey(1) in [ord('q'), 27]:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eye tracking")
    parser.add_argument('--source', type=str, default='Video.mp4', help='Video source (camera index, RTSP URL, or video file path)')
    parser.add_argument('--width', type=int, default=640, help='Width of the camera frame')
    parser.add_argument('--height', type=int, default=360, help='Height of the camera frame')
    parser.add_argument('--model_x', type=str, default="./v3_checkpoints/2024_02_21_13_14_56/best_model_0.001.pth", help='Path to the model for x prediction')
    parser.add_argument('--model_y', type=str, default="./v3_checkpoints/2024_02_21_13_14_56/best_model_0.001.pth", help='Path to the model for y prediction')
    parser.add_argument('--smoothing_factor', type=float, default=0.2, help='Smoothing factor for mouse movement')
    args = parser.parse_args()
    main(args)