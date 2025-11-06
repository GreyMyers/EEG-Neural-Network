import os
import argparse
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from functions import split_data, load_data, preprocess_raw_eeg, ACTIONS
from neural_nets import EEGNet

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # shuts down GPU

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def fit_and_save(model, epochs, train_X, train_y, validation_X, validation_y, batch_size):
    """
    Train the EEG model with early stopping and automatic checkpoint saving.
    """

    # --- Callbacks ---
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath='models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # --- Compile model ---
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    # --- Fit the model ---
    history = model.fit(
        train_X, train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_X, validation_y),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # --- Plot accuracy and loss curves ---
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n Training complete. Best model saved to models/best_model.keras")


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
    ch_names = [ch for ch in range(train_X.shape[1])]

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


# -------------------------------------------------------
# Train and save two EEG models: motor and frontal
# -------------------------------------------------------

def train_region(region_name, dataset_dir, chans, labels, model_path, epochs=300, batch_size=8):
    """
    Train an EEGNet model for a specific brain region.
    """
    print(f"\n============================")
    print(f"Training {region_name.upper()} Model")
    print(f"============================")

    # --- Dynamically update ACTIONS for this region ---
    from functions import ACTIONS
    ACTIONS.clear()
    ACTIONS.extend(labels)
    print(f"Using ACTIONS: {ACTIONS}")

    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("pictures", exist_ok=True)

    # --- Split data (train/val/test) ---
    split_data(starting_dir=dataset_dir, shuffle=True, splitting_percentage=(70, 20, 10))

    # --- Load data ---
    tmp_train_X, train_y = load_data(starting_dir=dataset_dir, shuffle=True, balance=True)
    tmp_validation_X, validation_y = load_data(starting_dir="validation_data", shuffle=True, balance=True)



    print("DEBUG tmp_train_X type:", type(tmp_train_X))
    print("DEBUG tmp_train_X shape:", np.array(tmp_train_X).shape)
    print("DEBUG example element shape:", np.array(tmp_train_X[0]).shape)

    # --- Preprocess raw EEG with task-specific bandpass ---
    if region_name.lower() == "motor":
        lowcut, highcut = 8, 30
    else:
        lowcut, highcut = 4, 30

    train_X, _ = preprocess_raw_eeg(tmp_train_X, lowcut=lowcut, highcut=highcut, coi3order=0)
    validation_X, _ = preprocess_raw_eeg(tmp_validation_X, lowcut=lowcut, highcut=highcut, coi3order=0)

    # --- Reshape for Conv2D input ---
    train_X = train_X.reshape((len(train_X), chans, train_X.shape[2], 1))
    validation_X = validation_X.reshape((len(validation_X), chans, validation_X.shape[2], 1))

    # --- Build model ---
    model = EEGNet(nb_classes=len(labels), Chans=chans, Samples=250)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
    model.summary()

    # --- Train with callbacks ---
    # Extract directory and base filename from model_path
    model_dir = os.path.dirname(model_path)
    model_basename = os.path.basename(model_path)
    model_name_without_ext = os.path.splitext(model_basename)[0]
    model_ext = os.path.splitext(model_basename)[1]
    
    # Save all models with epoch numbers
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, f"{model_name_without_ext}_epoch{{epoch:03d}}{model_ext}"),
        monitor='val_accuracy',
        save_best_only=False,  # Save all models
        verbose=1,
        mode='max'
    )

    history = model.fit(
        train_X, train_y,
        validation_data=(validation_X, validation_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],  # Removed early_stop callback
        verbose=1
    )

    # --- Plot performance ---
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{region_name.capitalize()} Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{region_name.capitalize()} Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"pictures/{region_name}_training_curve.png")
    plt.close()

    print(f"\n{region_name.capitalize()} model saved to {model_path}\n")


def main():
    """
    Train two EEGNet models:
      1. Motor (C3,C4) → left/right
      2. Frontal (Fp1,Fp2) → forward/stop
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--user-name', type=str, default='default', help='User name to scope datasets and model outputs')
    args = parser.parse_args()
    name = args.user_name

    # Prefer datasets under datasets/<name>/ if present; otherwise fallback to defaults
    named_root = os.path.join("datasets", name)
    if os.path.isdir(os.path.join(named_root, 'motor')) and os.path.isdir(os.path.join(named_root, 'frontal')):
        MOTOR_DIR = os.path.join(named_root, "motor")
        FRONTAL_DIR = os.path.join(named_root, "frontal")
        print(f"[DATA] Using per-user datasets at {named_root}")
    else:
        MOTOR_DIR = os.path.join("datasets", "motor")
        FRONTAL_DIR = os.path.join("datasets", "frontal")
        print(f"[DATA] Using default datasets at datasets/motor and datasets/frontal")

    # User-specific model output directory
    user_models_dir = os.path.join("models", name)
    os.makedirs(user_models_dir, exist_ok=True)

    # -------- Motor Cortex Model --------
    train_region(
        region_name="motor",
        dataset_dir=MOTOR_DIR,
        chans=2,
        labels=["left", "right"],
        model_path=os.path.join(user_models_dir, "motor_model.keras"),
        epochs=300
    )

    # -------- Frontal Lobe Model --------
    train_region(
        region_name="frontal",
        dataset_dir=FRONTAL_DIR,
        chans=2,
        labels=["forward", "stop"],
        model_path=os.path.join(user_models_dir, "frontal_model.keras"),
        epochs=300
    )


if __name__ == "__main__":
    main()