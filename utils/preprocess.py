from ..nn.global_graph import np


def to_categorical(inputs):
    if inputs.ndim > 2:
        raise ValueError('only accept 1-d or 2-d inputs')
    # convert Y to one-hot encode
    # for example,merge (batch,1) to (batch,)
    if inputs.ndim == 2:
        if inputs.shape[-1] == 1:
            inputs = np.sum(inputs, axis=-1)
        else:
            raise ValueError('can not convert %s to one-hot vector' % (inputs.__class__))

    n_class = np.max(inputs)[0] + 1
    encoded_Y = np.eye(n_class)[inputs].reshape(-1, n_class)
    return encoded_Y


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    lengths = []
    for x in sequences:
        lengths.append(len(x))
    num_samples = len(sequences)
    if maxlen is None:
        maxlen = max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)

    for idx, s in enumerate(sequences):
        if not len(s):
            continue

        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('unknown truncating type!')

        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':
            x[idx, -len(trunc):] = trunc
        elif padding == 'post':
            x[idx, :len(trunc)] = trunc
        else:
            raise ValueError('unknown padding type!')
    return x
