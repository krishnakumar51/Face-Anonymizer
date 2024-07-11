import cv2
import mediapipe as mp


# read image
img = cv2.imread("image.jpg")
H, W , _ = img.shape


# detect face
mp_face_detection = mp.solutions.face_detection

# 0 works best for short range(2 metres)  && 1 wordss best at long range (5 metres)
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    print(out.detections)

    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x * W)
            y1 = int(y * H)
            w = int(w * W)
            h = int(h * H)

            img[y1: y1+h, x1: x1+w, :] = cv2.blur(img[y1: y1+h, x1: x1+w, :],(40,40) )

            cv2.imshow('img', img)
            cv2.waitKey(0)

# blur faces

# save image
cv2.imwrite("blurred_face.jpg",img)
