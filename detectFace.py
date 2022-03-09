import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
import dlib
from collections import Counter
import math
import time

from sklearn import metrics
from sklearn.svm import SVC

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# load images from folder to array
def read_images_from_folder(folder):
    image_list = []
    label_list = []
    print("[INFO] Detecting and realigning faces...")
    for file_path in os.listdir(folder):
        file_ext = os.path.splitext(file_path)[1]
        if file_ext in [".jpg", ".jpeg"]:
            image_path = os.path.join(folder, file_path)
            image = cv2.imread(image_path)

            if image is not None:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(image_gray, (550, 550))

                # make a copy of the image to draw on it
                temp = resized_image.copy()

                detector = dlib.get_frontal_face_detector()
                detections = detector(temp, 1)

                sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                faces = dlib.full_object_detections()
                for det in detections:
                    faces.append(sp(temp, det))

                if len(faces):
                    # Bounding box and eyes
                    bb = [i.rect for i in faces]
                    bb = [((i.left(), i.top()),
                           (i.right(), i.bottom())) for i in bb]  # Convert out of dlib format

                    right_eyes = [[face.part(i) for i in range(36, 42)] for face in faces]
                    right_eyes = [[(i.x, i.y) for i in eye] for eye in right_eyes]  # Convert out of dlib format

                    left_eyes = [[face.part(i) for i in range(42, 48)] for face in faces]
                    left_eyes = [[(i.x, i.y) for i in eye] for eye in left_eyes]  # Convert out of dlib format

                    for i in bb:
                        cv2.rectangle(temp, i[0], i[1], (255, 0, 0), 5)  # Bounding box

                    for eye in right_eyes:
                        cv2.rectangle(temp, (max(eye, key=lambda x: x[0])[0], max(eye, key=lambda x: x[1])[1]),
                                      (min(eye, key=lambda x: x[0])[0], min(eye, key=lambda x: x[1])[1]),
                                      (0, 0, 255), 5)
                        for point in eye:
                            cv2.circle(temp, (point[0], point[1]), 2, (0, 255, 0), -1)

                    for eye in left_eyes:
                        cv2.rectangle(temp, (max(eye, key=lambda x: x[0])[0], max(eye, key=lambda x: x[1])[1]),
                                      (min(eye, key=lambda x: x[0])[0], min(eye, key=lambda x: x[1])[1]),
                                      (0, 255, 0), 5)
                        for point in eye:
                            cv2.circle(temp, (point[0], point[1]), 2, (0, 0, 255), -1)

                    # convert to numpy array to calculate center
                    left_eyes = np.array(left_eyes)
                    left_eyes = left_eyes.reshape(-1, left_eyes.shape[-1])
                    left_eye_center = left_eyes.mean(axis=0).astype("int")
                    right_eyes = np.array(right_eyes)
                    right_eyes = right_eyes.reshape(-1, right_eyes.shape[-1])
                    right_eye_center = right_eyes.mean(axis=0).astype("int")

                    cv2.line(temp, (left_eye_center[0], left_eye_center[1]), (right_eye_center[0], right_eye_center[1]),
                             (255, 255, 255), 5)

                    dY = right_eye_center[1] - left_eye_center[1]
                    dX = right_eye_center[0] - left_eye_center[0]
                    angle = np.degrees(np.arctan2(dY, dX)) - 180

                    # get center of image then rotate and resize
                    h, w = temp.shape[:2]
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
                    face_float = np.float32(cropped_face)

                    image_list.append(face_float.flatten('C'))
                    # label_list.append(os.path.splitext(file_path)[0]) # print full image name
                    label_list.append(os.path.splitext(file_path)[0].split("_")[0])  # print only name

                else:
                    print(os.path.splitext(file_path)[0], "does not have a face")

    image_list = np.array(image_list)
    image_list = image_list.transpose()
    return image_list, label_list


def roc_auc_score_multiclass(actual_class, pred_class, model_name):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    fpr, tpr, thresh = ({} for i in range(3))
    curr_loop = 0

    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = metrics.roc_auc_score(new_actual_class, new_pred_class)
        roc_auc_dict[per_class] = roc_auc

        fpr[curr_loop], tpr[curr_loop], thresh[curr_loop] = metrics.roc_curve(new_actual_class, new_pred_class)
        plt.plot(fpr[curr_loop], tpr[curr_loop])
        curr_loop += 1

    plt.title("ROC Curve - " + model_name)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig("ROC Curve - " + model_name, dpi=300)

    return roc_auc_dict


def analysis_report(test, prediction, model_name):
    print(f"Accuracy : {metrics.accuracy_score(test, prediction)}")
    print(f"Macro Precision : {metrics.precision_score(test, prediction, zero_division=0, average='weighted')}")
    print(f"Macro Recall : {metrics.recall_score(test, prediction, zero_division=0, average='weighted')}")
    print(f"Macro f1 score : {metrics.f1_score(test, prediction, zero_division=0, average='weighted')}")
    print(f"Classification report : {metrics.classification_report(test, prediction, zero_division=0)}")
    print(roc_auc_score_multiclass(test, prediction, model_name))


if __name__ == '__main__':
    start_time = time.perf_counter()
    print("[INFO] Loading training data...")
    data, labels = read_images_from_folder("train")
    adjusted_data = data - data.mean(axis=1, keepdims=True)
    covariance_matrix = np.cov(adjusted_data)

    print("[INFO] Calculating eigenvectors and eigenvalues...")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # sort eigenvalues in descending order
    # then, sort the eigenvectors to correspond to the new eigenvalues order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    top_n_eigenvectors = eigenvectors[:, 0:1000]

    # perform inner product with top n eigenvectors and the adjusted data
    transformed_data = np.dot(top_n_eigenvectors.T, adjusted_data)

    # train SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(transformed_data.T, labels)

    print("[INFO] Loading test data...")
    # read test image
    test_data, test_labels = read_images_from_folder("test")
    adjusted_test_data = test_data - data.mean(axis=1, keepdims=True)

    # perform inner product with top n eigenvectors and the adjusted test data
    transformed_test_data = np.dot(top_n_eigenvectors.T, adjusted_test_data)

    test_data_length = len(test_labels)
    result_euclid_pred, result_knn_pred = ([] for i in range(2))
    correct_prediction = 0

    # k = square root of N (number of samples)
    k = math.floor(math.sqrt(len(labels)))

    recognition_threshold = 4500

    print("[INFO] Predicting results of NCC, KNN & SVM...")
    # SVM classifier
    result_svm_pred = model.predict(transformed_test_data.T)

    for i in range(test_data_length):
        # NCC classifier
        euclid_dist = np.linalg.norm(transformed_test_data.T[i] - transformed_data.T, axis=1)
        # print(euclid_dist) # print the individual distances

        min_dist_index = np.argmin(euclid_dist)
        euclidean_predicted_face = "unknown" if euclid_dist[min_dist_index] > recognition_threshold else labels[
            min_dist_index]
        result_euclid_pred.append(euclidean_predicted_face)

        # kNN classifier
        k_nearest_indices = np.argpartition(euclid_dist, k)[:k]
        counter = Counter([labels[i] for i in k_nearest_indices])
        kNN_predicted_face = "unknown" if euclid_dist[k_nearest_indices].mean() > 6000 else \
        counter.most_common()[0][0]
        result_knn_pred.append(kNN_predicted_face)


        # print([labels[i] for i in k_nearest_indices]) # print the nearest neighbors
        # print(euclid_dist[k_nearest_indices]) # print the distances of the nearest neighbors
        # print(euclid_dist[k_nearest_indices].mean())
        # print("Euclidean predicted face: {}\n"
        #       "KNN faces: {}\n"
        #       "Majority face (a): {}\n"
        #       "Actual face: {}\n".
        #       format(euclidean_predicted_face,
        #              [labels[i] for i in k_nearest_indices],
        #              counter.most_common()[0],
        #              test_labels[i]))

        print("Actual face: ", test_labels[i])
        print("Euclidean predicted face:", euclidean_predicted_face)
        print("KNN predicted face: ", kNN_predicted_face)
        print("SVM predicted face: ", result_svm_pred[i])

    print("\nEuclid Analysis \n------------------------------")
    analysis_report(test_labels, result_euclid_pred, "Euclid")

    print("\nKNN Analysis \n------------------------------")
    analysis_report(test_labels, result_knn_pred, "KNN")

    print("\nSVM Analysis \n------------------------------")
    analysis_report(test_labels, result_svm_pred, "SVM")
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"The program took {math.floor(total_time / 60)} minutes and {math.ceil(total_time % 60)} seconds to "
          f"complete.")

    #     if euclidean_predicted_face == test_labels[i]:
    #         correct_prediction += 1
    #
    # print("Correct predictions: {}/{}\nAccuracy: {}".
    #       format(correct_prediction, test_data_length, correct_prediction / test_data_length))

    # print(f"Accuracy: {round(accuracy_score(test_labels_arr, results_arr), 2)}")
    # print(f"Precision: {round(precision_score(test_labels_arr, results_arr), 2)}")
    # print(f"Recall: {round(recall_score(test_labels_arr, results_arr), 2)}")
    # print(f"F1_score: {round(f1_score(test_labels_arr, results_arr), 2)}")

    # # show average face
    # float_img = data.mean(axis=1, keepdims=True)
    # im = np.array(float_img, dtype=np.uint8)
    # im_unflatten = np.reshape(im, (100, 100))
    # cv2.imwrite('mean_image.jpg', im_unflatten)
    # cv2.imshow("mean_image", im_unflatten)
    # cv2.waitKey(0)
    #
    # # show each face - average face
    # for i in range(len(labels)):
    #     im = np.array(adjusted_data.T[i], dtype=np.uint8)
    #     im_unflatten = np.reshape(im, (100, 100))
    #     cv2.imwrite("adjusted_image.jpg", im_unflatten)
    #     cv2.imshow("adjusted_image", im_unflatten)
    #     cv2.waitKey(0)
