# ============================================
#  Real-Time EEG Control (Motor + Frontal)
# ============================================

from functions import preprocess_raw_eeg
from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from tensorflow import keras
from timeit import default_timer as timer
import numpy as np
import threading
import argparse
import time
import cv2


# -----------------------
# Shared Memory Object
# -----------------------
class Shared:
    def __init__(self):
        self.sample = None
        self.key = None


# -----------------------
# Graphical Interface
# -----------------------
class GraphicalInterface:
    def __init__(self, WIDTH=500, HEIGHT=500, SQ_SIZE=40, MOVE_SPEED=5):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.SQ_SIZE = SQ_SIZE
        self.MOVE_SPEED = MOVE_SPEED

        self.square = {
            'x1': int(WIDTH / 2 - SQ_SIZE / 2),
            'x2': int(WIDTH / 2 + SQ_SIZE / 2),
            'y1': int(HEIGHT / 2 - SQ_SIZE / 2),
            'y2': int(HEIGHT / 2 + SQ_SIZE / 2)
        }

        self.box = np.ones((SQ_SIZE, SQ_SIZE, 3)) * np.random.uniform(size=(3,))
        self.horizontal_line = np.ones((HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
        self.vertical_line = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))


# -----------------------
# EEG Acquisition Thread
# -----------------------
def acquire_signals():
    count = 0
    while True:
        with mutex:
            if count == 0:
                time.sleep(2)
                count += 1
            else:
                time.sleep(0.5)

            data = board.get_current_board_data(250)
            sample = []

            eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD)
            for channel in eeg_channels:
                sample.append(data[channel])

            shared_vars.sample = np.array(sample)

            if shared_vars.key == ord("q"):
                break
        time.sleep(0.1)


# -----------------------
# EEG Processing + GUI
# -----------------------
def compute_signals():
    # --- Load trained models ---
    MOTOR_MODEL = "models/motor_model.keras"      # Trained on C3, C4 → left/right
    FRONTAL_MODEL = "models/frontal_model.keras"  # Trained on Fp1, Fp2 → forward/stop

    motor_model = keras.models.load_model(MOTOR_MODEL)
    frontal_model = keras.models.load_model(FRONTAL_MODEL)

    MOTOR_CH = [0, 1]     # C3, C4
    FRONTAL_CH = [2, 3]   # Fp1, Fp2

    EMA_motor = np.array([-1, -1])
    EMA_frontal = np.array([-1, -1])
    alpha = 0.4

    gui = GraphicalInterface()
    first_run = True
    count_down = 100

    while True:
        with mutex:
            if count_down == 0:
                gui = GraphicalInterface()
                count_down = 100

            env = np.zeros((gui.HEIGHT, gui.WIDTH, 3))

            if shared_vars.sample is not None and shared_vars.sample.shape[0] >= 4:
                # --- Motor Prediction (Left/Right) ---
                motor_data = shared_vars.sample[MOTOR_CH, :]
                motor_input, _ = preprocess_raw_eeg(
                    motor_data.reshape((1, 2, 250)), fs=250, lowcut=7, highcut=45, coi3order=0
                )
                motor_input = motor_input.reshape((1, 2, 250, 1))
                motor_out = motor_model.predict(motor_input, verbose=0)[0]

                if EMA_motor[0] == -1:
                    EMA_motor = motor_out.copy()
                else:
                    EMA_motor = alpha * motor_out + (1 - alpha) * EMA_motor

                motor_action = np.argmax(EMA_motor)  # 0=left, 1=right
                motor_conf = np.max(EMA_motor)

                # --- Frontal Prediction (Forward/Stop) ---
                frontal_data = shared_vars.sample[FRONTAL_CH, :]
                frontal_input, _ = preprocess_raw_eeg(
                    frontal_data.reshape((1, 2, 250)), fs=250, lowcut=7, highcut=45, coi3order=0
                )
                frontal_input = frontal_input.reshape((1, 2, 250, 1))
                frontal_out = frontal_model.predict(frontal_input, verbose=0)[0]

                if EMA_frontal[0] == -1:
                    EMA_frontal = frontal_out.copy()
                else:
                    EMA_frontal = alpha * frontal_out + (1 - alpha) * EMA_frontal

                frontal_action = np.argmax(EMA_frontal)  # 0=forward, 1=stop
                frontal_conf = np.max(EMA_frontal)

                # --- Movement Logic ---
                if motor_conf > 0.5:
                    if motor_action == 0:  # Left
                        gui.square['x1'] -= gui.MOVE_SPEED
                        gui.square['x2'] -= gui.MOVE_SPEED
                    elif motor_action == 1:  # Right
                        gui.square['x1'] += gui.MOVE_SPEED
                        gui.square['x2'] += gui.MOVE_SPEED

                if frontal_conf > 0.5:
                    if frontal_action == 0:  # Forward (Concentrated)
                        gui.square['y1'] -= gui.MOVE_SPEED
                        gui.square['y2'] -= gui.MOVE_SPEED
                    elif frontal_action == 1:  # Stop (Relaxed)
                        pass  # no vertical motion

                print(f"Motor: {['Left','Right'][motor_action]} ({motor_conf*100:.1f}%) | "
                      f"Frontal: {['Forward','Stop'][frontal_action]} ({frontal_conf*100:.1f}%)")

            # --- GUI rendering ---
            count_down -= 1
            env[:, gui.WIDTH // 2 - 5:gui.WIDTH // 2 + 5, :] = gui.vertical_line
            env[gui.HEIGHT // 2 - 5:gui.HEIGHT // 2 + 5, :, :] = gui.horizontal_line
            env[gui.square['y1']:gui.square['y2'], gui.square['x1']:gui.square['x2']] = gui.box

            cv2.imshow('EEG BCI', env)

            # --- FPS ---
            if first_run:
                first_run = False
                start = timer()
            else:
                end = timer()
                fps = 1 / max((end - start), 1e-6)
                print(f"\rFPS: {fps:.1f}", end='')
                start = timer()

            shared_vars.key = cv2.waitKey(1) & 0xFF
            if shared_vars.key == ord("q"):
                cv2.destroyAllWindows()
                break

        time.sleep(0.05)


# -----------------------
# Main Entry
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, default='COM3')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    # board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)  # use synthetic for testing
    board.prepare_session()

    shared_vars = Shared()
    mutex = threading.Lock()

    board.start_stream()

    acquisition = threading.Thread(target=acquire_signals)
    computing = threading.Thread(target=compute_signals)

    acquisition.start()
    computing.start()

    acquisition.join()
    computing.join()

    board.stop_stream()
    board.release_session()