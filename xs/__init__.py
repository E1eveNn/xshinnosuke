from utils.toolkit import gradient_check, SummaryProfile
from nn.rng import manual_seed_all, randn, tensor, zeros, ones, rand, randint
import nn


# dtype
float16 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.float16)
float32 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.float32)
float64 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.float64)
int8 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.int8)
uint8 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.uint8)
int16 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.int16)
uint16 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.uint16)
int32 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.int32)
uint32 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.uint32)
int64 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.int64)
uint64 = nn.GLOBAL.np.dtype(nn.GLOBAL.np.uint64)
bool = nn.GLOBAL.np.dtype(nn.GLOBAL.np.bool)


def cuda_available():
    try:
        import cupy
    except ImportError:
        return False
    return True
