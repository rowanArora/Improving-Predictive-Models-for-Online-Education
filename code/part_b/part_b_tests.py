"""Tests for Part B Model"""
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


def main() -> None:
    """
    Test the saved model.
    """
    autoencoder = load_model('part_b_model.h5')

    question_ids = np.array([793, 418, 1152])
    user_ids = np.array([102, 104, 256])
    subject_ids = np.array(["[0, 1, 10, 14, 184, 232]", "[0, 39, 42, 125]", "[0, 1, 12, 177]"])
    label_encoder = LabelEncoder()
    encoded_subject_ids = label_encoder.fit_transform(subject_ids)

    predictions = autoencoder.predict([question_ids, user_ids, encoded_subject_ids])
    binary_predictions = (predictions > 0.5).astype(int)

    print("Sample New Inputs:")
    print("User IDs:", user_ids)
    print("Question IDs:", question_ids)
    print("Subject IDs:", subject_ids)
    print("Predictions:", binary_predictions)


if __name__ == "__main__":
    main()
