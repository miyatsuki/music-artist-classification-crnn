from numpy.random import rand
import src.utility as utility
import src.models as models
import os
import numpy as np

# from tensorflow.keras.optimizers import Adam


def predict(
    nb_classes=20,
    slice_length=911,
    artist_folder="artists",
    song_folder="song_data",
    save_weights_folder="weights",
    # lr=0.0001,
    random_states=42,
):
    """
    Main function for training the model and testing
    """

    weights = os.path.join(
        save_weights_folder,
        str(nb_classes)
        + "_"
        + str(slice_length)
        + "_"
        + str(random_states)
        + "/checkpoint.ckpt",
    )

    print("Loading dataset...")
    (
        Y_train,
        X_train,
        S_train,
        Y_test,
        X_test,
        S_test,
        Y_val,
        X_val,
        S_val,
    ) = utility.load_dataset_song_split(
        song_folder_name=song_folder,
        artist_folder=artist_folder,
        nb_classes=nb_classes,
        random_state=random_states,
    )

    print("Loaded and split dataset. Slicing songs...")

    # Create slices out of the songs
    X_train, Y_train, S_train = utility.slice_songs(
        X_train, Y_train, S_train, length=slice_length
    )
    X_val, Y_val, S_val = utility.slice_songs(X_val, Y_val, S_val, length=slice_length)
    X_test, Y_test, S_test = utility.slice_songs(
        X_test, Y_test, S_test, length=slice_length
    )

    print("Training set label counts:", np.unique(Y_train, return_counts=True))

    # Encode the target vectors into one-hot encoded vectors
    Y_train, le, enc = utility.encode_labels(Y_train)
    Y_test, le, enc = utility.encode_labels(Y_test, le, enc)
    Y_val, le, enc = utility.encode_labels(Y_val, le, enc)

    # Reshape data as 2d convolutional tensor shape
    X_train = X_train.reshape(X_train.shape + (1,))
    X_val = X_val.reshape(X_val.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))

    # build the model
    # model = models.CRNN2D(X_train.shape, nb_classes=Y_train.shape[1])
    model = models.CRNN2D(X_train.shape, nb_classes=nb_classes)

    # model.compile(
    #     loss="categorical_crossentropy", optimizer=Adam(lr=lr), metrics=["accuracy"]
    # )
    # model.summary()

    # Load weights that gave best performance on validation set
    model.load_weights(weights)

    y_score = model.predict(X_test)
    print(y_score)
    print(S_test)
    print(le.classes_)

    with open("raw_score.csv", "w") as f:
        f.write("song" + "," + ",".join(le.classes_) + "\n")
        for S_raw, y_raw in zip(S_test, y_score):
            f.write(S_raw + "," + ",".join(str(y_col) for y_col in y_raw) + "\n")


if __name__ == "__main__":
    slice_len = 32
    random_states = 0

    predict(
        nb_classes=5,
        slice_length=slice_len,
        save_weights_folder="weights_song_split",
        random_states=random_states,
    )
