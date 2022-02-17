import numpy as np
import cv2
import os

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
                resized_image = cv2.resize(image, (100, 100))
                image_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                image_float = np.float32(image_gray)

                cv2.imshow('pic: ', image_gray)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                image_list.append(image_float.flatten('C'))
                # image_list.append(image_float.reshape(10000,))
                label_list.append(os.path.splitext(file_path)[0])

    image_list = np.array(image_list)
    image_list = image_list.transpose()
    return image_list, label_list

if __name__ == '__main__':
    data, labels = read_images_from_folder("faces")
    adjusted_data = data - data.mean(axis=1, keepdims=True)
    covariance_matrix = np.cov(adjusted_data.transpose())
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # sort eigenvalues in descending order
    # then, sort the eigenvectors to correspond to the new eigenvalues order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    top_n_eigenvectors = eigenvectors[:, :2]
    # k_eigenvectors = eigenvectors[:2, :]
    eigen_faces = top_n_eigenvectors.T.dot(adjusted_data.T)
    weights = adjusted_data.T.dot(eigen_faces.T)

    # perform inner product with top n eigenvectors and the adjusted data
    feature_vector = top_n_eigenvectors.transpose()
    transformed_data = np.inner(feature_vector, adjusted_data)
    transformed_data = transformed_data.transpose()

    # read test image
    test_data, test_data_label = read_images_from_folder("testface")
    adjusted_test_data = test_data - data.mean(axis=1, keepdims=True)

    test_data_length = len(test_data_label)
    for i in range(test_data_length):
        test_weight = adjusted_test_data.T[i].dot(eigen_faces.T)
        label_index = np.argmin(np.linalg.norm(test_weight - weights, axis=1))
        print("Predicted face: ", labels[label_index])
        print("Actual face: ", test_data_label[i])

    # perform inner product with top n eigenvectors and the adjusted test data
    # transformed_test_data = np.inner(feature_vector, adjusted_test_data)
    # transformed_test_data = transformed_test_data.transpose()

    # print(weights)

    #  show average face
    # float_img = data.mean(axis=1, keepdims=True)
    # im = np.array(adjusted_test_data, dtype=np.uint8)
    # im_unflatten = np.reshape(im, (100, 100))
    # cv2.imwrite('mean_image.jpg', im_unflatten)
    # cv2.imshow("mean_image", im_unflatten)
    # cv2.waitKey(0)