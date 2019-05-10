import numpy as np


def get_padded_window(x, genomic_position, n_positions, desired_window_size):
    center_position = genomic_position
    window = (desired_window_size - 1) // 2
    data = x[max(0, center_position - window): center_position + window + 1]
    if data.size != desired_window_size:
        right_flank = (center_position + window) - n_positions
        left_flank = (center_position - window)
        if right_flank >= 0:
            data = np.pad(data, (0, right_flank + 1), 'constant')
        else:
            data = np.pad(data, (np.abs(left_flank), 0), 'constant')
    return data


def sequential_data_generator(celltypes, assays, data, n_positions, batch_size, average_data, desired_window_size=2001):
    data_keys = []
    data_values = []
    for k, v in data.items():
        data_keys.append(k)
        data_values.append(v)
    start = 0
    indices_to_assay = {assays.index(assay): assay for _, assay in data_keys}
    indices = np.array(
        [
            [celltypes.index(celltype) for celltype, _ in data_keys],
            [assays.index(assay) for _, assay in data_keys]
        ]
    )
    tracks = data_values
    while True:
        celltype_idxs = np.zeros(batch_size, dtype='int32')
        assay_idxs = np.zeros(batch_size, dtype='int32')
        genomic_25bp_idxs = np.arange(start, start + batch_size) % n_positions
        genomic_250bp_idxs = genomic_25bp_idxs // 10
        genomic_5kbp_idxs = genomic_25bp_idxs // 200
        value = np.zeros(batch_size)
        average = np.zeros(batch_size * desired_window_size)
        idxs = np.random.randint(len(data), size=batch_size)
        for i, idx in enumerate(idxs):
            celltype_idxs[i] = indices[0, idx]
            assay_idxs[i] = indices[1, idx]
            value[i] = tracks[idx][genomic_25bp_idxs[i]]
            average[i * desired_window_size: (i + 1) * desired_window_size] = get_padded_window(
                average_data[indices_to_assay[indices[1, idx]]],
                genomic_25bp_idxs[i],
                n_positions,
                desired_window_size
            )
        d = {
            'celltype_input': celltype_idxs, 
            'assay_input': assay_idxs, 
            'genome_25bp_input': genomic_25bp_idxs, 
            'genome_250bp_input': genomic_250bp_idxs,
            'genome_5kbp_input': genomic_5kbp_idxs,
            'average_input': average.reshape(-1, desired_window_size)
        }
        yield d, value
        start += batch_size
