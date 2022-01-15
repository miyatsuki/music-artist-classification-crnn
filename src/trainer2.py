import os
import pathlib
from os.path import isfile

import dill
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import src.models as models
import src.utility as utility


def train_model(
    slice_length=911,
    song_folder="song_data",
    plots=True,
    save_metrics_folder="metrics",
    save_weights_folder="weights",
    batch_size=16,
    nb_epochs=200,
    early_stop=10,
    lr=0.0001,
    random_states=42,
):
    """
    Main function for training the model and testing
    """
    songs = list(pathlib.Path(song_folder).iterdir())
    songs = list(filter(lambda x: x.name != ".DS_Store", songs))
    nb_classes = len(songs)

    weights = os.path.join(
        save_weights_folder, f"{slice_length}_{random_states}/checkpoint.ckpt"
    )
    os.makedirs(save_weights_folder, exist_ok=True)
    os.makedirs(save_metrics_folder, exist_ok=True)

    print("Loading dataset...")
    spectrogram_raw = []
    song_name_raw = []
    for song in songs:
        with open(song, "rb") as fp:
            loaded_song = dill.load(fp)
            spectrogram_raw.append(loaded_song[1])
            song_name_raw.append(song)

    print("Loaded and split dataset. Slicing songs...")
    # Create empty lists for train and test sets
    spectrogram = []
    song_name = []

    # Create slices out of the songs
    for i, spec in enumerate(spectrogram_raw):
        slices = int(spec.shape[1] / slice_length)
        for j in range(slices - 1):
            spectrogram.append(spec[:, slice_length * j : slice_length * (j + 1)])
            song_name.append(song_name_raw[i])

    X = np.array(spectrogram)
    y = np.array(song_name)

    print("Training set label counts:", np.unique(y, return_counts=True))

    # Encode the target vectors into one-hot encoded vectors
    y, le, enc = utility.encode_labels(y)
    print("Label Encoder")
    print(le.inverse_transform(list(range(nb_classes))))

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, stratify=y
    )

    # Reshape data as 2d convolutional tensor shape
    X_train = X_train.reshape(X_train.shape + (1,))
    X_val = X_val.reshape(X_val.shape + (1,))

    # build the model
    model = models.CRNN2D(X_train.shape, nb_classes=nb_classes)
    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(lr=lr), metrics=["accuracy"]
    )
    model.summary()

    checkpointer = ModelCheckpoint(
        filepath=weights, verbose=1, save_best_only=True, save_weights_only=True
    )
    earlystopper = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=early_stop, verbose=0, mode="auto"
    )

    # Train the model
    print("Input Data Shape", X_train.shape)
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        epochs=nb_epochs,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[checkpointer, earlystopper],
    )
    if plots:
        utility.plot_history(history)
