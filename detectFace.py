import numpy as np
import cv2
import os

# load images from folder to array
def read_images_from_folder(folder):
    image_list = []
    label_list = []
    for file_path in os.listdir(folder):
        file_ext = os.path.splitext(file_path)[1]
        if file_ext in [".jpg", ".jpeg", ".heic"]:
            image_path = os.path.join(folder, file_path)
            image = cv2.imread(image_path)

            if image is not None:
                resized_image = cv2.resize(image, (100, 100))
                image_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                image_float = np.float32(image_gray)

                # cv2.imshow('pic: ', image_gray)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                image_list.append(image_float.flatten('C'))
                label_list.append(os.path.splitext(file_path)[0])

    image_list = np.array(image_list)
    image_list = image_list.transpose()
    return image_list, label_list

if __name__ == '__main__':
    data, labels = read_images_from_folder("faces")
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
    test_data, test_labels = read_images_from_folder("testface")
    adjusted_test_data = test_data - data.mean(axis=1, keepdims=True)

    # perform inner product with top n eigenvectors and the adjusted test data
    transformed_test_data = np.dot(top_n_eigenvectors.T, adjusted_test_data)

    test_data_length = len(test_labels)
    for i in range(test_data_length):
        euclid_dist = np.linalg.norm(transformed_test_data.T[i] - transformed_data.T, axis=1)
        print(euclid_dist)
        min_dist_index = np.argmin(euclid_dist)
        print("Predicted face: ", labels[min_dist_index])
        print("Actual face: ", test_labels[i])

    # # weights method
    # eigen_faces = top_n_eigenvectors.T.dot(adjusted_data.T)
    # weights = adjusted_data.T.dot(eigen_faces.T)

    # for i in range(test_data_length):
    #     test_weight = adjusted_test_data.T[i].dot(eigen_faces.T)
    #     label_index = np.argmin(np.linalg.norm(test_weight - weights, axis=1))
    #     print("Predicted face: ", labels[label_index])
    #     print("Actual face: ", test_labels[i])

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