import cv2
import torch
import numpy as np
from PIL import Image
import os
from face_recognition_model import FaceRecognitionModel
from torchvision import transforms

class RealtimeFaceRecognition:
    def __init__(self):
        print("Initializing Face Recognition System...")
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the trained model
        try:
            self.model = FaceRecognitionModel().to(self.device)
            model_path = 'best_model.pth'
            
            if os.path.exists(model_path):
                print(f"\nFound model file: {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Model loaded successfully!")
            else:
                raise Exception(f"Model file not found: {model_path}")
                
            self.model.eval()
            
        except Exception as e:
            print(f"\nError loading model: {e}")
            raise

        # Load dataset for student recognition
        self.student_data = {}
        dataset_path = "dataset"
        if os.path.exists(dataset_path):
            for student_folder in os.listdir(dataset_path):
                if student_folder.startswith('student_'):
                    student_id = student_folder.split('_')[1]
                    self.student_data[student_id] = student_folder
            print(f"\nFound {len(self.student_data)} students in dataset")
            print("Student IDs:", list(self.student_data.keys()))

        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Set confidence threshold
        self.confidence_threshold = 0.5
        self.process_this_frame = True
        self.face_names = []
        
        print("\nSystem initialization complete!")
        print("\nControls:")
        print("- Press '+' to increase confidence threshold")
        print("- Press '-' to decrease confidence threshold")
        print("- Press 'q' to quit")
        print("- Press 's' to save frame")

    def process_face(self, face_img):
        """Process face image and get prediction"""
        try:
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_img)
            
            # Transform image
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probability = torch.sigmoid(output).item()
            return probability
            
        except Exception as e:
            print(f"Error processing face: {e}")
            return 0.0

    def run_detection(self):
        print("Starting real-time detection...")
        
        video_capture = cv2.VideoCapture(0)
        frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            if self.process_this_frame:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                self.face_names = []
                for (x, y, w, h) in faces:
                    try:
                        # Extract face ROI
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Get probability
                        probability = self.process_face(face_img)
                        
                        # Determine if recognized
                        is_recognized = probability > self.confidence_threshold
                        if is_recognized:
                            name = f"Student 1 ({probability:.2f})"  # For first client
                        else:
                            name = f"Unknown ({probability:.2f})"
                        
                        self.face_names.append((name, (x, y, w, h)))
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue

            self.process_this_frame = not self.process_this_frame

            # Display the results
            for name, (x, y, w, h) in self.face_names:
                color = (0, 255, 0) if "Student" in name else (0, 0, 255)
                
                # Draw box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label
                label_y = y - 10 if y - 10 > 10 else y + h + 10
                cv2.putText(frame, name, (x, label_y),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # Display threshold
            cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Face Recognition', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                print(f"Threshold increased to: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.0, self.confidence_threshold - 0.05)
                print(f"Threshold decreased to: {self.confidence_threshold:.2f}")
            elif key == ord('s'):
                frame_count += 1
                cv2.imwrite(f'test_frame_{frame_count}.jpg', frame)
                print(f"Saved frame {frame_count}")

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        face_recognizer = RealtimeFaceRecognition()
        face_recognizer.run_detection()
    except Exception as e:
        print(f"Error: {str(e)}") 