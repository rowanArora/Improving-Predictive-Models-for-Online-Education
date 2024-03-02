"""Part B:"""
from utils import *
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
embedding_dim = 128
dropout_rate = 0.3
learning_rate = 0.01
batch_size = 256
epochs = 10


def load_data(base_path="../data") -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """ Load and merge the data as Pandas DataFrame.

    :return: (train_data, valid_data, test_data)
        WHERE:
        train_data: pd.DataFrame containing training data.
        valid_data: pd.DataFrame containing validation data.
        test_data: pd.DataFrame containing test data.
    """
    # Load and explode the metadata we need.
    question_md_df = pd.read_csv('../data/question_meta.csv')
    subject_ids = question_md_df['subject_id']
    label_encoder = LabelEncoder()
    encoded_subject_ids = label_encoder.fit_transform(subject_ids)
    question_md_df['subject_id'] = encoded_subject_ids
    # Load training dataset.
    train = pd.read_csv('../data/train_data.csv')
    # Load validation dataset.
    valid = pd.read_csv('../data/valid_data.csv')
    # Load testing dataset.
    test = pd.read_csv('../data/test_data.csv')

    # Set the common column names.
    col_1 = 'question_id'

    # Merge the metadata with each dataset.
    train_data = pd.merge(train, question_md_df, on=col_1)
    valid_data = pd.merge(valid, question_md_df, on=col_1)
    test_data = pd.merge(test, question_md_df, on=col_1)

    return train_data, valid_data, test_data


def extract_features(train_data: pd.DataFrame,
                     valid_data: pd.DataFrame,
                     test_data: pd.DataFrame) -> \
        (list, list, list, pd.Series, pd.Series, pd.Series):
    """ Extract the features we will build the model upon.

    :return: (X_train, X_valid, X_test, y_train, y_valid, y_test)
        WHERE:
        X_train: list of training dataset features.
        X_valid: list of validation dataset features.
        X_test: list of testing dataset features.
        y_train: Series of "is_correct" from training dataset.
        y_valid: Series of "is_correct" from validation dataset.
        y_test: Series of "is_correct" from testing dataset.
    """
    X_train = [train_data['user_id'], train_data['question_id'], train_data['subject_id']]
    X_valid = [valid_data['user_id'], valid_data['question_id'], valid_data['subject_id']]
    X_test = [test_data['user_id'], test_data['question_id'], test_data['subject_id']]

    y_train = train_data['is_correct']
    y_valid = valid_data['is_correct']
    y_test = test_data['is_correct']

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_num_features(train_data: pd.DataFrame) -> (int, int, int):
    """ Get the number of unique students, questions, and subjects.
    """
    num_users = len(train_data['user_id'])
    num_questions = len(train_data['question_id'])
    num_subjects = len(train_data['subject_id'])

    return num_users, num_questions, num_subjects


def build_autoencoder(num_users: int, num_questions: int, num_subjects: int) -> Model:
    """ Build the model based on the features extracted.
    """
    question_input = Input(shape=(1,))
    user_input = Input(shape=(1,))
    subject_input = Input(shape=(1,))

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
    question_embedding = Embedding(input_dim=num_questions, output_dim=embedding_dim)(question_input)
    subject_embedding = Embedding(input_dim=num_subjects, output_dim=embedding_dim)(subject_input)

    merged = Concatenate()([question_embedding, user_embedding, subject_embedding])
    x = Dense(embedding_dim, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(8, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(4, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(2, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(4, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(8, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    decoded = Dense(embedding_dim, activation='relu')(x)

    output_layer = Dense(1, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=[question_input, user_input, subject_input], outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate),
                        loss='mse',
                        metrics=['accuracy'])

    return autoencoder


def train_autoencoder(autoencoder: Model,
                      X_train: list,
                      y_train: pd.Series,
                      X_valid: list,
                      y_valid: pd.Series,
                      epochs: int,
                      batch_size: int):
    """ Train the model on the training data and validate it on the validation data.
    """
    class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                      classes=np.unique(y_train), 
                                                      y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = autoencoder.fit(X_train,
                              y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(X_valid, y_valid),
                              callbacks=[early_stopping],
                              class_weight=class_weight_dict)
    return history


def evaluate_autoencoder(autoencoder: Model, X_test: list, y_test: pd.Series):
    """ Evaluate the accuracy of the model.
    """
    loss, accuracy = autoencoder.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy * 100:.4f}%")


def main() -> None:
    """
    Run everything.
    """
    train_data, valid_data, test_data = load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = extract_features(train_data, valid_data, test_data)
    num_users, num_questions, num_subjects = get_num_features(train_data)
    autoencoder = build_autoencoder(num_users, num_questions, num_subjects)
    history = train_autoencoder(autoencoder, X_train, y_train, X_valid, y_valid, epochs, batch_size)
    successful_epochs = len(history.history['loss'])
    evaluate_autoencoder(autoencoder, X_test, y_test)
    autoencoder.save('part_b_model.h5')

    fig = plt.subplot()
    fig.plot(np.arange(successful_epochs), history.history['val_accuracy'], color='red', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend('upper right')
    plt.title('Validation Accuracy Per Epoch')
    plt.savefig('part_b.png')


if __name__ == "__main__":
    main()
