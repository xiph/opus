
def clip_to_int16(x):
    int_min = -2**15
    int_max = 2**15 - 1
    x_clipped = max(int_min, min(x, int_max))
    return x_clipped
