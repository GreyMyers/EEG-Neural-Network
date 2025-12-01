from sklearn.model_selection import KFold
from functions import split_data, standardize, load_data, preprocess_raw_eeg, ACTIONS
from neural_nets import cris_net, res_net, TA_CSPNN, EEGNet
import os

from sklearn.model_selection import KFold, cross_val_score
from matplotlib import pyplot as plt

from tensorflow import keras
import tensorflow as tf
import numpy as np
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # shuts down GPU

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def fit_and_save(model, epochs, train_X, train_y, validation_X, validation_y, batch_size):
    # fits the network epoch by epoch and saves only accurate models

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(epochs):
        history = model.fit(train_X, train_y, epochs=1, batch_size=batch_size,
                            validation_data=(validation_X, validation_y))

        train_acc.append(history.history["accuracy"][-1])
        train_loss.append(history.history["loss"][-1])
        val_acc.append(history.history["val_accuracy"][-1])
        val_loss.append(history.history["val_loss"][-1])

        MODEL_NAME = f"models/grey/{round(val_acc[-1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(val_loss[-1], 2)}.keras"

        if round(val_acc[-1] * 100, 4) >= 77 and round(train_acc[-1] * 100, 4) >= 77:
            # saving only relevant models
            model.save(MODEL_NAME)
            print("saved: ", MODEL_NAME)

    # Create combined plot with accuracy and loss after training completes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy subplot
    ax1.plot(np.arange(len(val_acc)), val_acc, label='val', linewidth=2)
    ax1.plot(np.arange(len(train_acc)), train_acc, label='train', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Loss subplot
    ax2.plot(np.arange(len(val_loss)), val_loss, label='val', linewidth=2)
    ax2.plot(np.arange(len(train_loss)), train_loss, label='train', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("pictures/motor_training_curves.png", dpi=150)
    plt.show()


def kfold_cross_val(model, train_X, train_y, epochs, num_folds, batch_size):
    acc_per_fold = []
    loss_per_fold = []

    kfold = KFold(n_splits=num_folds, shuffle=True, )
    fold_no = 1

    for train, test in kfold.split(train_X, train_y):
        model = model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        history = model.fit(train_X[train], train_y[train],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0)
        scores = model.evaluate(train_X[test], train_y[test], verbose=0)

        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]};'
              f' {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        fold_no += 1

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {str(loss_per_fold[i])[:4]} - Accuracy: {str(acc_per_fold[i])[:4]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {str(np.mean(acc_per_fold))[:4]} (+- {str(np.std(acc_per_fold))[:4]})')
    print(f'> Loss: {str(np.mean(loss_per_fold))[:4]}')
    print('------------------------------------------------------------------------')


def check_other_classifiers(train_X, train_y, test_X, test_y):
    from pyriemann.classification import MDM, TSclassifier
    from sklearn.linear_model import LogisticRegression
    from pyriemann.estimation import Covariances
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP
    import seaborn as sns
    import pandas as pd

    train_y = [np.where(i == 1)[0][0] for i in train_y]
    test_y = [np.where(i == 1)[0][0] for i in test_y]

    cov_data_train = Covariances().transform(train_X)
    cov_data_test = Covariances().transform(test_X)
    cv = KFold(n_splits=10, random_state=42)
    clf = TSclassifier()
    scores = cross_val_score(clf, cov_data_train, train_y, cv=cv, n_jobs=1)
    print("Tangent space Classification accuracy: ", np.mean(scores))

    clf = TSclassifier()
    clf.fit(cov_data_train, train_y)
    print(clf.score(cov_data_test, test_y))

    mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
    scores = cross_val_score(mdm, cov_data_train, train_y, cv=cv, n_jobs=1)
    print("MDM Classification accuracy: ", np.mean(scores))
    mdm = MDM()
    mdm.fit(cov_data_train, train_y)

    fig, axes = plt.subplots(1, 2)
    ch_names = [ch for ch in range(8)]

    df = pd.DataFrame(data=mdm.covmeans_[0], index=ch_names, columns=ch_names)
    g = sns.heatmap(
        df, ax=axes[0], square=True, cbar=False, xticklabels=2, yticklabels=2)
    g.set_title('Mean covariance - feet')

    df = pd.DataFrame(data=mdm.covmeans_[1], index=ch_names, columns=ch_names)
    g = sns.heatmap(
        df, ax=axes[1], square=True, cbar=False, xticklabels=2, yticklabels=2)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    g.set_title('Mean covariance - hands')

    # dirty fix
    plt.sca(axes[0])
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.savefig("meancovmat.png")
    plt.show()


def main():
    # Setup paths for Grey dataset
    DATASET_DIR = os.path.join("datasets", "grey", "motor")
    MODELS_DIR = os.path.join("models", "grey")
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("pictures", exist_ok=True)
    
    split_data(starting_dir=DATASET_DIR, shuffle=True, splitting_percentage=(70, 20, 10), division_factor=0, coupling=False)

    # loading dataset
    tmp_train_X, train_y = load_data(starting_dir="training_data", shuffle=True, balance=True)
    tmp_validation_X, validation_y = load_data(starting_dir="validation_data", shuffle=True, balance=True)

    print(f"Train samples: {len(tmp_train_X)}, Val samples: {len(tmp_validation_X)}")

    # cleaning the raw data (bandpass 8-30 Hz for motor imagery)
    train_X, fft_train_X = preprocess_raw_eeg(tmp_train_X, lowcut=8, highcut=30, coi3order=0)
    validation_X, fft_validation_X = preprocess_raw_eeg(tmp_validation_X, lowcut=8, highcut=30, coi3order=0)

    # check_other_classifiers(train_X, train_y, validation_X, validation_y)

    # reshaping
    train_X = train_X.reshape((len(train_X), len(train_X[0]), len(train_X[0, 0]), 1))
    validation_X = validation_X.reshape((len(validation_X), len(validation_X[0]), len(validation_X[0, 0]), 1))

    # computing absolute value element-wise of the ffts, necessary if crisnet is chosen
    # fft_train_X = standardize(np.abs(fft_train_X))[:, :, :, np.newaxis]
    # fft_validation_X = standardize(np.abs(fft_validation_X))[:, :, :, np.newaxis]

    """"
    Start from here if you want to try ConvLSTM2D as first layers in the networks
    
    n_subseq = 10
    n_timesteps = 25
    # train_X = train_X.reshape((len(train_X), n_subseq, len(train_X[0]), n_timesteps, 1))
    # validation_X = validation_X.reshape((len(validation_X), n_subseq, len(validation_X[0]), n_timesteps, 1))
    """

    print("train_X shape: ", train_X.shape)

    # Build model for 8 channels, 250 samples
    # model = TA_CSPNN(nb_classes=len(ACTIONS), Timesamples=250, Channels=8,
    #                timeKernelLen=50, dropOut=0.3, Ft=11, Fs=6)

    model = EEGNet(nb_classes=len(ACTIONS), Chans=8, Samples=250)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    keras.utils.plot_model(model, "pictures/net.png", show_shapes=True)

    batch_size = 32
    epochs = 150

    # kfold_cross_val(model, train_X, train_y, epochs, num_folds=10, batch_size=batch_size)
    fit_and_save(model, epochs, train_X, train_y, validation_X, validation_y, batch_size)


if __name__ == "__main__":
    main()