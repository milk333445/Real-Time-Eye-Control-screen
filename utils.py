import cv2
import time
import numpy as np
import copy
import os
import pyautogui

def capture_eye_region(frame, x_min, x_max, y_min, y_max):
    
    x_min, y_min = max(0, x_min-10), max(0, y_min-10)
    x_max, y_max = min(frame.shape[1], x_max+10), min(frame.shape[0], y_max+10)
    eye_region = frame[y_min:y_max, x_min:x_max]
    
    return eye_region


def save_eye_image(eye_region, folder="eye_images", file_name="eye.jpg"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, file_name)
    cv2.imwrite(path, eye_region)
    print(f"Saved {file_name} to {folder} folder.")
    
def get_relative_mouse_position(wScr, hScr):
    mouse_x, mouse_y = pyautogui.position()
    return mouse_x / wScr, mouse_y / hScr


def create_eye_region_mask(frame, x_min, x_max, y_min, y_max):
    eye_region = capture_eye_region(frame, x_min, x_max, y_min, y_max)
    h, w, _ = frame.shape
    eye_mask = np.full((h, w, 3), 255, dtype=np.uint8)
    eye_region_height, eye_region_width, _ = eye_region.shape
    eye_mask[y_min:y_min+eye_region_height, x_min:x_min+eye_region_width] = eye_region
    return eye_mask

def resize_and_pad_image(image, target_size):
    
    h, w = image.shape[:2]
    if h == w:
        return cv2.resize(image, target_size)
    longer_side = max(h, w)
    
    top = (longer_side - h) // 2 
    bottom = longer_side - h - top 
    left = (longer_side - w) // 2 
    right = longer_side - w - left 
    
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    resize_image = cv2.resize(padded_image, target_size)
    return resize_image


class SmoothMouseMover:
    def __init__(self, smoothing_factor=0.2):
        self.smoothing_factor = smoothing_factor
        self.current_x = 0
        self.current_y = 0

    def update_position(self, new_x, new_y):
        self.current_x = self.current_x * (1 - self.smoothing_factor) + new_x * self.smoothing_factor
        self.current_y = self.current_y * (1 - self.smoothing_factor) + new_y * self.smoothing_factor
        return self.current_x, self.current_y