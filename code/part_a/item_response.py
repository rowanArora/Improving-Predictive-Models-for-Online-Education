import random

from utils import *

import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.

    for i in range(len(data['user_id'])):
        student = data['user_id'][i]
        question = data['question_id'][i]
        is_correct = data['is_correct'][i]

        inner = theta[student] - beta[question]
        s_inner = sigmoid(inner)
        log_lklihood += is_correct * np.log(s_inner) + (1 - is_correct) * np.log(1 - s_inner)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    update_theta = np.zeros(len(np.unique(data['user_id'])))
    update_beta = np.zeros(len(np.unique(data['question_id'])))

    for i in range(len(data['user_id'])):
        student = data['user_id'][i]
        question = data['question_id'][i]
        is_correct = data['is_correct'][i]

        inner = theta[student] - beta[question]
        s_inner = sigmoid(inner)
        update_theta[student] += s_inner - is_correct
        update_beta[question] += is_correct - s_inner

    theta -= lr * update_theta
    beta -= lr * update_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(len(np.unique(data['user_id'])))
    beta = np.zeros(len(np.unique(data['question_id'])))

    train_acc_lst = []
    val_acc_lst = []

    train_neg_lld_lst = []
    val_neg_lld_lst = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_acc = evaluate(data=data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_acc = evaluate(data=val_data, theta=theta, beta=beta)

        train_acc_lst.append(train_acc)
        train_neg_lld_lst.append(train_neg_lld)
        val_acc_lst.append(val_acc)
        val_neg_lld_lst.append(val_neg_lld)

        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_lst, train_neg_lld_lst, val_acc_lst, val_neg_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    learning_rate = 0.01
    iterations = 150
    theta, beta, train_acc_lst, train_neg_lld_lst, val_acc_lst, val_neg_lld_lst = irt(train_data, val_data,
                                                                                      learning_rate, iterations)

    # Part (b):
    fig = plt.subplot()
    fig.plot(np.arange(iterations), train_neg_lld_lst, color='blue', label='Training Log-Likelihoods')
    fig.plot(np.arange(iterations), val_neg_lld_lst, color='red', label='Validation Log-Likelihoods')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Log-Likelihood Per Iteration')
    plt.savefig('q2b.png')
    plt.close()

    # Part (c):
    final_val_acc = evaluate(data=val_data, theta=theta, beta=beta)
    print(f"Final Validation Accuracy: {final_val_acc}")
    final_test_acc = evaluate(data=test_data, theta=theta, beta=beta)
    print(f"Final Test Accuracy: {final_test_acc}")

    # Part (d):
    questions = random.choices(np.arange(len(np.unique(train_data['user_id']))), k=3)
    min_theta = min(theta)
    max_theta = max(theta)
    theta_range = np.linspace(min_theta, max_theta, 150)
    colors = ['blue', 'red', 'green']
    fig = plt.subplot()
    plt.xlabel('Theta')
    plt.ylabel('Probability of the Correct Response')
    plt.title('Probability of the Correct Response Given for a Question')
    for i in range(3):
        curr_q = questions[i]
        probabilities = sigmoid(theta_range - beta[curr_q])
        fig.plot(theta_range, probabilities, color=colors[i], label=f"Question {curr_q}")
    plt.legend(loc='lower right')
    plt.savefig('q2c.png')


if __name__ == "__main__":
    main()
