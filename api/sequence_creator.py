import numpy as np


# note: only feed with features (NOT labels!)
def create_sequence_3d(
        features: np.ndarray,
        n_sequence: int) -> (np.ndarray, [int]):
    """
    create 3D Sequence for RNN/LSTM
    endpoints are used to filter valid sequence
    """
    output_size = (features.shape[0] - (n_sequence - 1), n_sequence, features.shape[1])
    output = np.full(output_size, np.nan)

    for i in range(n_sequence):
        ix_end = features.shape[0] - (n_sequence - 1) + i
        output[:, i] = features[i:ix_end]

    return output
