from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import welch
import numpy as np
from scipy.signal import filtfilt, cheby1

def prepare_board():
                  
    # Initialize board parameters
    params = BrainFlowInputParams()
    params.serial_port = 'COM3'

    # Create a board objet
    board_id = BoardIds.CYTON_BOARD.value
    board = BoardShim(board_id, params)

    # Prepare the board
    BoardShim.enable_dev_board_logger()  # Enables logging for debugging
    try:
        board.prepare_session()  # Prepares the board for data acquisition
        status = 'Board is ready and connected!'
    except Exception as e:
        print('Error: ', e)
        status = e
    return board,board_id,status