"""Thread-block layout constants used by the launchers and the TileLang kernels."""

import torch

# On AMD, a CU's 64 KB of shared memory holds one thread block of 4096 points, and with 8 points per
# thread that block has two wavefronts per SIMD instead of one.
FFT_POINTS_PER_THREAD = 8 if torch.version.hip else 16
# Batch rows per thread block of the chunk scans, which is the M extent of a tensor-core tile.
CHUNK_ROWS = 16
# Columns of a chunk scan's tensor-core accumulator that are this far apart are in the same thread.
CHUNK_COLUMN_STRIDE = 16 if torch.version.hip else 8
# Rows per thread block of `kernels.final_state`, which is the M extent of a tensor-core tile.
FINAL_STATE_ROWS = 16
