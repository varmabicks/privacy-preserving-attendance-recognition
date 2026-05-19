import cv2
import os
import numpy as np
from PIL import Image
import time

class FaceCapture:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.images_per_angle = 25
        self.frame_skip = 10
        self.min_face_size = (150, 150)
        
        # Sequential angle states
        self.angles = [
            {
                'name': 'Front Face',
                'count': 0,
                'instructions': ['Look straight at camera', 'Maintain neutral expression']
            },
            {
                'name': 'Left Side',
                'count': 0,
                'instructions': ['Turn head slightly left', 'Keep eyes level']
            },
            {
                'name': 'Right Side',
                'count': 0,
                'instructions': ['Turn head slightly right', 'Keep eyes level']
            },
            {
                'name': 'Up Angle',
                'count': 0,
                'instructions': ['Tilt head slightly up', 'Keep eyes on camera']
            },
            {
                'name': 'Down Angle',
                'count': 0,
                'instructions': ['Tilt head slightly down', 'Keep eyes on camera']
            },
            {
                'name': 'Expressions',
                'count': 0,
                'instructions': ['Show slight smile', 'Blink normally']
            }
        ]
        self.current_angle_index = 0
        self.capturing = False
        self.waiting_for_enter = True

    def verify_quality(self, face_img):
        """Verify face image quality"""
        if face_img.shape[0] < self.min_face_size[0] or face_img.shape[1] < self.min_face_size[1]:
            return False
            
        brightness = np.mean(face_img)
        if brightness < 40 or brightness > 250:
            return False
            
        contrast = np.std(face_img)
        if contrast < 20:
            return False
            
        laplacian = cv2.Laplacian(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if laplacian < 100:
            return False
            
        return True

    def draw_interface(self, frame, faces):
        """Draw interface elements on frame"""
        height, width = frame.shape[:2]
        
        # Draw guide box
        guide_size = (400, 400)
        x = (width - guide_size[0]) // 2
        y = (height - guide_size[1]) // 2
        cv2.rectangle(frame, (x, y), (x + guide_size[0], y + guide_size[1]), (0, 255, 0), 2)
        
        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw instructions
        y_offset = 30
        if not self.capturing:
            instructions = [
                "Press 'S' to start capture process",
                "Press 'Q' to quit"
            ]
        else:
            current_angle = self.angles[self.current_angle_index]
            if self.waiting_for_enter:
                instructions = [
                    f"Prepare for: {current_angle['name']}",
                    *current_angle['instructions'],
                    "Press ENTER when ready"
                ]
            else:
                instructions = [
                    f"Capturing: {current_angle['name']}",
                    f"Progress: {current_angle['count']}/{self.images_per_angle}",
                    *current_angle['instructions']
                ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw overall progress
        total_images = sum(angle['count'] for angle in self.angles)
        total_required = self.images_per_angle * len(self.angles)
        progress = f"Total Progress: {total_images}/{total_required}"
        cv2.putText(frame, progress, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def capture_faces(self, student_id):
        """Main capture function"""
        print("\nStarting face capture for Student", student_id)
        
        cap = cv2.VideoCapture(0)
        frame_count = 0
        
        # Create directory for student
        save_dir = os.path.join('dataset', f'student_{student_id}')
        os.makedirs(save_dir, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue
                
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_face_size
            )
            
            # Draw interface
            self.draw_interface(frame, faces)
            
            # Handle face capture
            if self.capturing and not self.waiting_for_enter and len(faces) > 0:
                current_angle = self.angles[self.current_angle_index]
                if current_angle['count'] < self.images_per_angle:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    
                    if self.verify_quality(face_img):
                        img_name = os.path.join(save_dir, 
                            f'face_{current_angle["name"]}_{current_angle["count"]}.jpg')
                        cv2.imwrite(img_name, face_img)
                        current_angle['count'] += 1
                        time.sleep(0.2)  # Add delay between captures
                else:
                    self.current_angle_index += 1
                    if self.current_angle_index < len(self.angles):
                        self.waiting_for_enter = True
                    else:
                        break
            
            cv2.imshow('Face Capture', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not self.capturing:
                self.capturing = True
            elif key == 13 and self.waiting_for_enter:  # Enter key
                self.waiting_for_enter = False
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\nCapture Summary:")
        for angle in self.angles:
            print(f"{angle['name']}: {angle['count']}/{self.images_per_angle} images")
        
        total_images = sum(angle['count'] for angle in self.angles)
        print(f"\nTotal images captured: {total_images}")
        return total_images

    def verify_dataset(self):
        """Verify captured images"""
        print("\nVerifying captured images...")
        total_verified = 0
        
        for student_folder in os.listdir('dataset'):
            if student_folder.startswith('student_'):
                folder_path = os.path.join('dataset', student_folder)
                image_files = [f for f in os.listdir(folder_path) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))]
                total_verified += len(image_files)
                print(f"{student_folder}: {len(image_files)} images")
        
        print(f"\nTotal verified images: {total_verified}")
        return total_verified

if __name__ == "__main__":
    student_id = input("Enter student ID: ")
    print("\nFace Capture Instructions:")
    print("1. Press 'S' to start the capture process")
    print("2. Follow on-screen instructions")
    print("3. Press ENTER to start capturing each angle")
    print("4. Press 'Q' to quit at any time\n")
    
    face_capture = FaceCapture()
    total_captured = face_capture.capture_faces(student_id)
    face_capture.verify_dataset() 