from dataclasses import dataclass


@dataclass
class IndexConfig:
    d: int
    nlist: int
    M: int
    nbits: int = 8
    metric: str = "L2"            # "L2" | "IP"
    train_size: int | None = None  # None = use all vectors
    niter: int = 10               # k-means iterations for coarse quantizer + PQ
    chunk_size: int = 500_000     # vectors per add() call
    opq: bool = False             # OPQ rotation; enable for high-dim uneven embeddings
    hnsw_m: int = 0              # HNSW neighbors per node; 0 = flat quantizer, 32 = HNSW32

    def __post_init__(self):
        if self.d % self.M != 0:
            raise ValueError(f"d={self.d} must be divisible by M={self.M}")
        if self.metric not in ("L2", "IP"):
            raise ValueError(f"metric must be 'L2' or 'IP', got '{self.metric}'")
        if self.nlist < 1:
            raise ValueError(f"nlist must be >= 1, got {self.nlist}")
        if self.train_size is not None and self.train_size < self.nlist:
            raise ValueError(
                f"train_size={self.train_size} < nlist={self.nlist}; "
                "need at least one training vector per centroid"
            )
