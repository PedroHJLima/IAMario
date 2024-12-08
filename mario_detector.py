import cv2
import numpy as np
from time import time
import os

class MarioActionDetector:
    def __init__(self):
        self.prev_y = None  
        self.jump_count = 0
        self.start_y = None 
        self.min_y = None 
        self.is_jumping = False
        template_path = os.path.join(os.path.dirname(__file__), 'mario_template.png')
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise Exception(f"Não foi possível carregar o template do Mario em {template_path}")

    def detect_mario(self, frame):
        frame_display = frame.copy()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_skin = np.array([10, 100, 100])  
        upper_skin = np.array([30, 255, 255]) 
        
        lower_clothes = np.array([10, 30, 100]) 
        upper_clothes = np.array([30, 150, 255])
        
        lower_red1 = np.array([0, 100, 100])  
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])  
        upper_red2 = np.array([180, 255, 255])
        
        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
        mask_clothes = cv2.inRange(hsv, lower_clothes, upper_clothes)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mario_mask = cv2.bitwise_or(mask_skin, mask_clothes)
        mario_mask = cv2.bitwise_or(mario_mask, mask_red)
        
        kernel = np.ones((2,2), np.uint8)
        mario_mask = cv2.morphologyEx(mario_mask, cv2.MORPH_OPEN, kernel)
        mario_mask = cv2.morphologyEx(mario_mask, cv2.MORPH_CLOSE, kernel)
        
        mario_contours, _ = cv2.findContours(mario_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if mario_contours:
            mario_contour = max(mario_contours, key=cv2.contourArea)
            if cv2.contourArea(mario_contour) > 50:
                x, y, w, h = cv2.boundingRect(mario_contour)
                
                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if self.prev_y is not None:
                    moving_up = y < self.prev_y - 5
                    moving_down = y > self.prev_y + 5
                    
                    if moving_up and not self.is_jumping:
                        self.is_jumping = True
                        self.start_y = self.prev_y
                        self.min_y = y
                    
                    elif self.is_jumping and not moving_down:
                        self.min_y = min(y, self.min_y) if self.min_y is not None else y
                    
                    elif self.is_jumping and moving_down:
                        if (self.start_y is not None and 
                            self.min_y is not None and 
                            self.start_y - self.min_y > 30):
                            self.jump_count += 1
                        self.is_jumping = False
                        self.start_y = None
                        self.min_y = None
                
                self.prev_y = y
                
                jump_height = self.start_y - self.min_y if self.start_y and self.min_y else 0
                state = "PULANDO" if self.is_jumping else "PARADO"
                cv2.putText(frame_display, f"{state} Alt: {jump_height}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.putText(frame_display, f'Pulos: {self.jump_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        debug_masks = np.hstack((mask_skin, mask_clothes, mask_red, mario_mask))
        cv2.imshow('Debug (Skin | Clothes | Red | Combined)', debug_masks)
        
        return frame_display

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = MarioActionDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = detector.detect_mario(frame)
        
        cv2.imshow('Mario Jump Counter', processed_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f'\nTotal de pulos detectados: {detector.jump_count}')

if __name__ == "__main__":
    video_path = "mario.mp4"
    process_video(video_path)
