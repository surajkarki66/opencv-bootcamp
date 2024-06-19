import os
import cv2
import argparse
import mediapipe as mp


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process images, videos, or webcam feed for face detection and blurring."
    )
    parser.add_argument(
        "--mode", default="webcam", help="Mode of operation: webcam, video or image"
    )
    parser.add_argument(
        "--filePath", default=None, help="Path to the video or image file"
    )
    return parser.parse_args()


def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1 = int(bbox.xmin * W), int(bbox.ymin * H)
            w, h = int(bbox.width * W), int(bbox.height * H)

            # Ensure the bounding box is within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            # Ensure width and height are positive
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = cv2.blur(img[y1:y2, x1:x2], (50, 50))
            img[y1 : y1 + h, x1 : x1 + w] = cv2.blur(
                img[y1 : y1 + h, x1 : x1 + w], (50, 50)
            )

    return img


def main():
    args = parse_arguments()
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5
    ) as face_detection:
        if args.mode == "image" and args.filePath:
            img = cv2.imread(args.filePath)
            img = process_img(img, face_detection)
            cv2.imwrite(os.path.join(output_dir, "output.png"), img)
            print("Output saved to output/output.png")

        elif args.mode == "video" and args.filePath:
            cap = cv2.VideoCapture(args.filePath)
            if not cap.isOpened():
                print(f"Error: Could not open video file {args.filePath}")
                return
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read video file {args.filePath}")
                return

            output_video = cv2.VideoWriter(
                os.path.join(output_dir, "output.mp4"),
                cv2.VideoWriter_fourcc(*"MP4V"),
                25,
                (frame.shape[1], frame.shape[0]),
            )

            while ret:
                frame = process_img(frame, face_detection)
                output_video.write(frame)
                ret, frame = cap.read()

            cap.release()
            output_video.release()

        elif args.mode == "webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from webcam")
                    break

                frame = process_img(frame, face_detection)
                cv2.imshow("Webcam Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
