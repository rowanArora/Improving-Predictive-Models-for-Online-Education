from matplotlib import pyplot as plt


from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        out = F.sigmoid(self.g(out))
        out = F.sigmoid(self.h(out))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    training_losses = []
    validation_accuracies = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            reg_term = lamb / 2.0 * model.get_weight_norm()

            loss = torch.sum((output - target) ** 2.) + reg_term
            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        validation_accuracies.append(valid_acc)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        training_losses.append(train_loss)

    return training_losses, validation_accuracies

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    k_values = [10]
    best_k = None
    best_valid_acc = 0.0
    best_lr = None
    best_lamb = None

    lr_values = [0.1]
    num_epoch = 40
    lamb_values = [0.001, 0.01, 0.1, 1.0]

    training_costs_dict = {}
    validation_accuracies_dict = {}

    for k in k_values:
        for lr in lr_values:
            for lamb in lamb_values:
                print(f"\nTraining with k={k}, learning rate={lr}, lambda={lamb}")
                model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
                training_losses, validation_accuracies = train(
                    model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch
                )

                training_costs_dict[f'k={k}, lr={lr}, lambda={lamb}'] = training_losses
                validation_accuracies_dict[f'k={k}, lr={lr}, lambda={lamb}'] = validation_accuracies

                valid_acc = validation_accuracies[-1]

                print(f"Validation Accuracy: {valid_acc}")

                # Check if the current combination is better than the previous best
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_k = k
                    best_lr = lr
                    best_lamb = lamb

    print(f"\nBest k: {best_k}, Best lr: {best_lr}, Best lamb: {best_lamb} Best Validation Accuracy: {best_valid_acc}")

    plt.figure(figsize=(10, 6))

    for key, training_costs in training_costs_dict.items():
        plt.plot(range(1, num_epoch + 1), training_costs, label=f'{key} - Training Cost')

    plt.xlabel('Epoch')
    plt.ylabel('Training Cost')
    plt.legend()
    plt.title('Training Costs for Different Hyperparameter Configurations')
    plt.show()

    plt.figure(figsize=(10, 6))

    for key, validation_accuracies in validation_accuracies_dict.items():
        plt.plot(range(1, num_epoch + 1), validation_accuracies, label=f'{key} - Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.title('Validation Accuracies for Different Hyperparameter Configurations')
    plt.show()

    best_model = AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
    training_losses, _ = train(best_model, lr=0.01, lamb=0.0, train_data=train_matrix,
                               zero_train_data=zero_train_matrix, valid_data=test_data,
                               num_epoch=num_epoch)

    test_acc = evaluate(best_model, zero_train_matrix, test_data)
    print(f"Final Test Accuracy with k={best_k}: {test_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
