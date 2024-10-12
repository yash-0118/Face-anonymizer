import os
import logging
import cv2
import mediapipe as mp
import argparse

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (only show errors)

# Suppress Mediapipe logs
logging.getLogger('mediapipe').setLevel(logging.ERROR)  # Suppress Mediapipe logs (only show errors)

def process_img(img, face_detection):
    if img is None:
        print("Error: Image not found or unable to read.")
        return img  # Return early if the image is None

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

            # Blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img

args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)
args = args.parse_args()

output_dir = r'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode in ["image"]:
        img = cv2.imread(args.filePath)
        if img is None:
            print("Error: Could not read the image. Please check the file path.")
            exit()

        img = process_img(img, face_detection)
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read the video file. Please check the file path.")
            exit()

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))

        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
        img_count = 0  # Counter for saved images
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Capture and save photo
                img_count += 1
                img_path = os.path.join(output_dir, f'captured_image_{img_count}.png')
                cv2.imwrite(img_path, frame)
                print(f"Image saved: {img_path}")

            elif key == ord('q'):  # Exit the loop if 'q' is pressed
                break

            # cv2.waitKey(25)
            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()  # Close all OpenCV windows
