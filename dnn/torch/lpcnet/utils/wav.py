import wave

def wavwrite16(filename, x, fs):
    """ writes x as int16 to file with name filename

        If x.dtype is int16 x is written as is. Otherwise,
        it is scaled by 2**15 - 1 and converted to int16.
    """
    if x.dtype != 'int16':
        x = ((2**15 - 1) * x).astype('int16')

    with wave.open(filename, 'wb') as f:
        f.setparams((1, 2, fs, len(x), 'NONE', ""))
        f.writeframes(x.tobytes())