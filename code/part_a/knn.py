from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("User-based Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    transposed_matrix = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    imputed_transposed_matrix = nbrs.fit_transform(transposed_matrix)
    imputed_matrix = imputed_transposed_matrix.T
    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)
    print("Item-based Validation Accuracy: {}".format(acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    val_user_accuracies = {}
    val_item_accuracies = {}
    for k in k_values:
        print("k = {}".format(k))
        val_user_accuracies[k] = knn_impute_by_user(sparse_matrix, val_data, k)
        val_item_accuracies[k] = knn_impute_by_item(sparse_matrix, val_data, k)

    plt.plot(k_values, list(val_user_accuracies.values()), color='blue', label='User Accuracies')
    plt.plot(k_values, list(val_item_accuracies.values()), color='red', label='Item Accuracies')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy (User)')
    plt.legend(loc='lower right')
    plt.show()

    best_user_k = max(val_user_accuracies, key=val_user_accuracies.get)
    print("best_user_k = {}".format(best_user_k))   
    best_item_k = max(val_item_accuracies, key=val_item_accuracies.get)
    print("best_item_k = {}".format(best_item_k))
    knn_impute_by_user(sparse_matrix, test_data, best_user_k)
    knn_impute_by_item(sparse_matrix, test_data, best_item_k)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
