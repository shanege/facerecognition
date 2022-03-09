import numpy as np
import cv2
import os
import dlib

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def read_images_from_folder(folder):
    current_name = ""

    for file_path in os.listdir(folder):
        file_ext = os.path.splitext(file_path)[1]
        if file_ext in [".jpg", ".jpeg"]:
            image_path = os.path.join(folder, file_path)
            image = cv2.imread(image_path)

            if image is not None:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(image_gray, (550, 550))
                temp = resized_image.copy()

                detector = dlib.get_frontal_face_detector()
                detections = detector(temp, 1)

                predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                faces = dlib.full_object_detections()
                for det in detections:
                    faces.append(predictor(temp, det))

                if len(faces):
                    # Bounding box and eyes
                    # bb = [i.rect for i in faces]
                    # bb = [((i.left(), i.top()),
                    #        (i.right(), i.bottom())) for i in bb]  # Convert out of dlib format

                    right_eyes = [[face.part(i) for i in range(36, 42)] for face in faces]
                    right_eyes = [[(i.x, i.y) for i in eye] for eye in right_eyes]  # Convert out of dlib format

                    left_eyes = [[face.part(i) for i in range(42, 48)] for face in faces]
                    left_eyes = [[(i.x, i.y) for i in eye] for eye in left_eyes]  # Convert out of dlib format

                    # for i in bb:
                    #     cv2.rectangle(temp, i[0], i[1], (255, 0, 0), 5)  # Bounding box
                    #
                    # for eye in right_eyes:
                    #     cv2.rectangle(temp, (max(eye, key=lambda x: x[0])[0], max(eye, key=lambda x: x[1])[1]),
                    #                   (min(eye, key=lambda x: x[0])[0], min(eye, key=lambda x: x[1])[1]),
                    #                   (0, 0, 255), 5)
                    #     for point in eye:
                    #         cv2.circle(temp, (point[0], point[1]), 2, (0, 255, 0), -1)
                    #
                    # for eye in left_eyes:
                    #     cv2.rectangle(temp, (max(eye, key=lambda x: x[0])[0], max(eye, key=lambda x: x[1])[1]),
                    #                   (min(eye, key=lambda x: x[0])[0], min(eye, key=lambda x: x[1])[1]),
                    #                   (0, 255, 0), 5)
                    #     for point in eye:
                    #         cv2.circle(temp, (point[0], point[1]), 2, (0, 0, 255), -1)

                    left_eyes = np.array(left_eyes)
                    left_eyes = left_eyes.reshape(-1, left_eyes.shape[-1])
                    left_eye_center = left_eyes.mean(axis=0).astype("int")

                    right_eyes = np.array(right_eyes)
                    right_eyes = right_eyes.reshape(-1, right_eyes.shape[-1])
                    right_eye_center = right_eyes.mean(axis=0).astype("int")

                    # cv2.line(temp, (left_eye_center[0], left_eye_center[1]), (right_eye_center[0], right_eye_center[1]), (255, 255, 255), 5)

                    dY = right_eye_center[1] - left_eye_center[1]
                    dX = right_eye_center[0] - left_eye_center[0]
                    angle = np.degrees(np.arctan2(dY, dX)) - 180

                    h, w = resized_image.shape[:2]
                    center = (w // 2, h // 2)

                    M = cv2.getRotationMatrix2D(center, angle, 1)

                    rotated = cv2.warpAffine(resized_image, M, (w, h))

                    rotated = cv2.resize(rotated, (100, 100))
                    # cv2.imshow("rotated", rotated)
                    # cv2.waitKey(0)

                    # find and crop the face from the aligned image
                    face = face_cascade.detectMultiScale(rotated, 1.1, 5)
                    for (x, y, w, h) in face:
                        if w >= h:
                            h = w
                        else:
                            w = h
                        cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 255, 0), 1)

                    cropped_face = rotated[y + 1:y + h, x + 1:x + w]
                    cropped_face = cv2.resize(cropped_face, (100, 100))

                    cv2.imshow("cropped", cropped_face)
                    cv2.waitKey(0)

                else:
                    print(os.path.splitext(file_path)[0], "does not have a face")

if __name__ == '__main__':
    # read_images_from_folder("tilted")
    # read_images_from_folder("test")
    # read_images_from_folder("train")
    read_images_from_folder("masked")
