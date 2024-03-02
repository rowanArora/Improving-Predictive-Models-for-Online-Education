from utils import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np


def main():
    # Load data
    train_data = load_train_csv("../data")
    valid_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    ensemble_predictions_valid = np.zeros(len(valid_data["is_correct"]))
    ensemble_predictions_test = np.zeros(len(test_data["is_correct"]))

    num_base_models = 3

    for i in range(num_base_models):
        bootstrap_sample = np.random.choice(len(train_data["is_correct"]), len(train_data["is_correct"]), replace=True)
        if i == 0:
            base_model = DecisionTreeClassifier()
        elif i == 1:
            base_model = RandomForestClassifier()
        else:
            base_model = GradientBoostingClassifier()

        base_model.fit(X=np.array([train_data["user_id"][j] for j in bootstrap_sample]).reshape(-1, 1),
                       y=np.array([train_data["is_correct"][j] for j in bootstrap_sample]))

        predictions_valid = base_model.predict(np.array(valid_data["user_id"]).reshape(-1, 1))
        predictions_test = base_model.predict(np.array(test_data["user_id"]).reshape(-1, 1))

        ensemble_predictions_valid += predictions_valid
        ensemble_predictions_test += predictions_test

    ensemble_predictions_valid /= num_base_models
    ensemble_predictions_test /= num_base_models

    threshold = 0.5
    valid_accuracy = evaluate(valid_data, ensemble_predictions_valid, threshold)
    test_predictions = [1 if p >= threshold else 0 for p in ensemble_predictions_test]
    test_accuracy = evaluate(test_data, ensemble_predictions_test, threshold)

    print(f"Validation Accuracy: {valid_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
