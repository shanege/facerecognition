import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
from math import sin, cos, radians

from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn import metrics


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

# load images from folder to array
def read_images_from_folder(folder):
    image_list = []
    label_list = []
    for file_path in os.listdir(folder):
        file_ext = os.path.splitext(file_path)[1]
        if file_ext in [".jpg", ".jpeg"]:
            image_path = os.path.join(folder, file_path)
            image = cv2.imread(image_path)

            if image is not None:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(image_gray, (100, 100))

                # detect the face and eyes in the image
                face = face_cascade.detectMultiScale(resized_image, 1.03, 5)

                if len(face):
                    # crop out face
                    for (x, y, w, h) in face:
                        if w >= h:
                            h = w
                        else:
                            w = h
                        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 255), 1)
                        # +1 to remove the border of the rectangle when cropping
                        face = resized_image[y + 1:y + h, x + 1:x + w]
                        resized_face = cv2.resize(face, (100, 100))
                        face_float = np.float32(resized_face)
                        # cv2.imshow('img', resized_face)
                        # cv2.waitKey()

                else:
                    for angle in range(-40, 40):
                        rimg = rotate_image(resized_image, angle)
                        face = face_cascade.detectMultiScale(rimg, 1.03, 5)
                        if len(face):
                            for (x, y, w, h) in face:
                                if w >= h:
                                    h = w
                                else:
                                    w = h
                            cv2.rectangle(rimg, (x, y), (x + w, y + h), (255, 255, 255), 1)
                            # +1 to remove the border of the rectangle when cropping
                            face = rimg[y + 1:y + h, x + 1:x + w]
                            resized_face = cv2.resize(face, (100, 100))
                            face_float = np.float32(resized_face)
                            # cv2.imshow('img', resized_face)
                            # cv2.waitKey()
                            break

                        else:
                            face_float = np.float32(resized_image)

                image_list.append(face_float.flatten('C'))
                # label_list.append(os.path.splitext(file_path)[0]) # print full image name
                label_list.append(os.path.splitext(file_path)[0].split("_")[0])  # print only name

    image_list = np.array(image_list)
    image_list = image_list.transpose()
    return image_list, label_list


if __name__ == '__main__':
    data, labels = read_images_from_folder("faces_r")
    adjusted_data = data - data.mean(axis=1, keepdims=True)
    covariance_matrix = np.cov(adjusted_data)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # sort eigenvalues in descending order
    # then, sort the eigenvectors to correspond to the new eigenvalues order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    top_n_eigenvectors = eigenvectors[:, 0:1000]

    # perform inner product with top n eigenvectors and the adjusted data
    transformed_data = np.dot(top_n_eigenvectors.T, adjusted_data)

    # read test image
    test_data, test_labels = read_images_from_folder("testface_r")
    adjusted_test_data = test_data - data.mean(axis=1, keepdims=True)

    # perform inner product with top n eigenvectors and the adjusted test data
    transformed_test_data = np.dot(top_n_eigenvectors.T, adjusted_test_data)

    test_data_length = len(test_labels)
    print(test_labels)
    print(labels)
    results = []
    correct_prediction = 0
    for i in range(test_data_length):
        euclid_dist = np.linalg.norm(transformed_test_data.T[i] - transformed_data.T, axis=1)
        # print(euclid_dist) # print the individual distances
        min_dist_index = np.argmin(euclid_dist)
        if euclid_dist[min_dist_index] > 4500:
            predicted_face = "unknown"

        else:
            predicted_face = labels[min_dist_index]

        results.append(predicted_face)
        print("Predicted face: {}\nActual face: {}\n".format(predicted_face, test_labels[i]))

        if predicted_face == test_labels[i]:
            correct_prediction += 1

    print("Correct predictions: {}/{}\nAccuracy: {}".format(correct_prediction, test_data_length,
                                                            correct_prediction / test_data_length))

    results_arr = np.array(results)
    test_labels_arr = np.array(test_labels)
    # print(f"Accuracy: {round(accuracy_score(test_labels_arr, results_arr), 2)}")
    # print(f"Precision: {round(precision_score(test_labels_arr, results_arr), 2)}")
    # print(f"Recall: {round(recall_score(test_labels_arr, results_arr), 2)}")
    # print(f"F1_score: {round(f1_score(test_labels_arr, results_arr), 2)}")
    print(metrics.classification_report(test_labels_arr, results_arr, zero_division=0))

    # # show average face
    # float_img = data.mean(axis=1, keepdims=True)
    # im = np.array(float_img, dtype=np.uint8)
    # im_unflatten = np.reshape(im, (100, 100))
    # cv2.imwrite('mean_image.jpg', im_unflatten)
    # cv2.imshow("mean_image", im_unflatten)
    # cv2.waitKey(0)

    # # show each face - average face
    # for i in range(len(labels)):
    #     im = np.array(adjusted_data.T[i], dtype=np.uint8)
    #     im_unflatten = np.reshape(im, (100, 100))
    #     cv2.imwrite("adjusted_image.jpg", im_unflatten)
    #     cv2.imshow("adjusted_image", im_unflatten)
    #     cv2.waitKey(0)
