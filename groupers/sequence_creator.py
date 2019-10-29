import numpy as np

# input format is 2D matrix
# axis0 contains samples
# axis1 contains sensors
# see also diagram DIAG01


def create_sequence_offset_matrix(xxs: np.ndarray, n_sequence: int):
    """
    map each element of array into
    array containing last n_sequence values (including value of current position)
    """
    n_past_values = n_sequence - 1
    output_size = (xxs.shape[0] - n_past_values, n_sequence, xxs.shape[1])
    output = np.full(output_size, np.nan)

    # Diagram DG-SEQ-PRE
    data_3d = xxs.reshape((xxs.shape[0], 1, xxs.shape[1]))

    # creating sequence column by column
    # we start with column for time step: t-n_sequence
    # and end with column t-0
    for i in range(n_sequence):
        # i = offset = column_group
        ix_end = xxs.shape[0] - n_past_values + i
        output[:, i, :] = data_3d[i:ix_end, 0, :]

    return output
