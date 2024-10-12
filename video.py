import os
import cv2
import mediapipe as mp
import argparse

def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blur the detected face
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))

    return img

# Parse arguments
args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')
args = args.parse_args()

# Output directory
output_dir = r'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)  # Open the webcam

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
        fps = 20.0  # Frames per second
        frame_width = int(cap.get(3))  # Frame width
        frame_height = int(cap.get(4))  # Frame height
        output_path = os.path.join(output_dir, 'output_video.mp4')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = process_img(frame, face_detection)  # Process and blur faces
            cv2.imshow('Webcam', frame)  # Display the frame

            # Write the frame to the video file
            out.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Exit the loop if 'q' is pressed
                break

        # Release the VideoCapture and VideoWriter objects
        cap.release()
        out.release()
        cv2.destroyAllWindows()  # Close all OpenCV windows
        print(f"Video saved at {output_path}")
