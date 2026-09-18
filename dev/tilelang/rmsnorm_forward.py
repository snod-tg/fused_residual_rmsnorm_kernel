"""
rmsnorm forward kernel apply by tilelang
"""

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import torch

"""
Inputs:
    x: Tensor([B, T, C], float16)
    w: Tnespr([C], float16)
    N: int # B * T

Output:
    mean2: Tensor([B, T], float32)
    y: Tensor([B, T, C], float32)

Definition:

    y =  w * x * rsqrt(x.pow(2).mean) 
"""

def ref_rmsnorm(x, w, eps=1e-5):
    return w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)

# my version
@tilelang.jit
def tl_rmsnorm_forward(X, W, BLOCK_N: int, eps: float):
    N, C = T.const("N, C")
    io_dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((N, C), io_dtype)
    W: T.Tensor((C,), io_dtype)
    Y = T.empty((N, C), io_dtype)
    mean2 = T.empty((N,), accum_dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as pid_n:
        X_local = T.alloc_fragment((BLOCK_N, C), io_dtype)
        W_local = T.alloc_shared((C,), io_dtype)
        X2_local = T.alloc_fragment((BLOCK_N, C), accum_dtype)
        Y_local = T.alloc_fragment((BLOCK_N, C), io_dtype)
        mean2_local = T.alloc_fragment((BLOCK_N,), accum_dtype)
        tmp_sum = T.alloc_fragment((BLOCK_N,), accum_dtype)

        T.copy(X[pid_n * BLOCK_N, :], X_local)
        T.copy(W, W_local)

        for i, j in T.Parallel(BLOCK_N, C):
            X2_local[i, j] = X_local[i, j].astype(accum_dtype) * X_local[i, j].astype(accum_dtype)

        T.reduce_sum(X2_local, tmp_sum, dim=1, clear=True)

        for i in T.Parallel(BLOCK_N):
            mean2_local[i] = tmp_sum[i] / C

        for i, j in T.Parallel(BLOCK_N, C):
            Y_local[i, j] = W_local[j].astype(accum_dtype) * X_local[i, j].astype(accum_dtype) * T.rsqrt(mean2_local[i] + eps)

        T.copy(mean2_local, mean2[pid_n * BLOCK_N])
        T.copy(Y_local, Y[pid_n * BLOCK_N, :])

    return Y, mean2

@tilelang.jit
def tl_rmsnorm_forward_splitc(X, W, BLOCK_N: int, BLOCK_C: int, eps: float):
    N, C = T.const("N, C")
    io_dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((N, C), io_dtype)
    W: T.Tensor((C,), io_dtype)
    Y = T.empty((N, C), io_dtype)
    mean2 = T.empty((N,), accum_dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as pid_n:
        X_local = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        W_local = T.alloc_shared((BLOCK_C,), io_dtype)
        X2_local = T.alloc_fragment((BLOCK_N, BLOCK_C), accum_dtype)
        Y_local = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        mean2_local = T.alloc_fragment((BLOCK_N,), accum_dtype)
        tmp_sum = T.alloc_fragment((BLOCK_N,), accum_dtype)

        # T.copy(X[pid_n * BLOCK_N, :], X_local)
        # T.copy(W, W_local)
        num_c_step = T.ceildiv(C, BLOCK_C)
        T.clear(X2_local) 
        for k in T.Serial(num_c_step):
            T.copy(X[pid_n * BLOCK_N, k * BLOCK_C], X_local)
            for i, j in T.Parallel(BLOCK_N, BLOCK_C):
                X2_local[i, j] += X_local[i, j].astype(accum_dtype) * X_local[i, j].astype(accum_dtype)

        T.reduce_sum(X2_local, tmp_sum, dim=1, clear=True) # 

        for i in T.Parallel(BLOCK_N):
            mean2_local[i] = tmp_sum[i] / C

        for k in T.Serial(num_c_step):
            kk = num_c_step - 1 - k  
            T.copy(X[pid_n * BLOCK_N, kk * BLOCK_C], X_local)
            T.copy(W[kk * BLOCK_C], W_local)
            for i, j in T.Parallel(BLOCK_N, BLOCK_C):
                Y_local[i, j] = W_local[j].astype(accum_dtype) * X_local[i, j].astype(accum_dtype) * T.rsqrt(mean2_local[i] + eps)

            T.copy(Y_local, Y[pid_n * BLOCK_N, kk * BLOCK_C])

        T.copy(mean2_local, mean2[pid_n * BLOCK_N])
        
    return Y, mean2


# ref version, without weight
@tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
def rms_norm_splitk(A, blk_m, blk_k):
    M, N = T.const("M, N")
    dtype = T.float

    A: T.Tensor((M, N), dtype)
    B = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
        A_shared = T.alloc_shared((blk_m, blk_k), dtype)
        A_local = T.alloc_fragment((blk_m, blk_k), dtype)
        A_powsum = T.alloc_fragment((blk_m,), dtype)

        num_k_step = T.ceildiv(N, blk_k)
        T.clear(A_local)
        for k in T.Serial(num_k_step):
            T.copy(A[bx * blk_m, k * blk_k], A_shared)
            for i, j in T.Parallel(blk_m, blk_k):
                A_local[i, j] += A_shared[i, j] * A_shared[i, j]
        T.reduce_sum(A_local, A_powsum, dim=1)
        for i in T.Parallel(blk_m):
            A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)

        for k in T.Serial(num_k_step):
            # reverse, better cache hit rate
            T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
            for i, j in T.Parallel(blk_m, blk_k):
                A_shared[i, j] *= A_powsum[i]
            T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return B


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
def rms_norm(A, blk_m):
    M, N = T.const("M, N")
    dtype = T.float

    A: T.Tensor((M, N), dtype)
    B = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
        A_local = T.alloc_fragment((blk_m, N), dtype)
        A_pow_local = T.alloc_fragment((blk_m, N), dtype)
        A_powsum = T.alloc_fragment((blk_m,), dtype)

        T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_local)
        for i, j in T.Parallel(blk_m, N):
            A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
        T.reduce_sum(A_pow_local, A_powsum, dim=1)
        for i in T.Parallel(blk_m):
            A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)
        for i, j in T.Parallel(blk_m, N):
            A_local[i, j] *= A_powsum[i]
        T.copy(A_local, B[bx * blk_m : (bx + 1) * blk_m, :])

    return B


# ═══════════════════════════════════════════════════════════════
#  测试: RMSNorm forward  (整行版 vs 分块版)
# ═══════════════════════════════════════════════════════════════


def check(y, mean2, x, w, eps, tag=""):
    """校验 y (以及可选的 mean2)"""
    ref_y = ref_rmsnorm(x, w, eps).to(torch.float16)
    torch.testing.assert_close(y, ref_y, rtol=1e-2, atol=1e-2)

    if mean2 is not None:
        # mean2 的定义 = mean(x^2), 是个 (N,) 的 fp32
        ref_mean2 = x.float().pow(2).mean(dim=-1)
        torch.testing.assert_close(mean2, ref_mean2, rtol=1e-2, atol=1e-2)

    suffix = "y + mean2" if mean2 is not None else "y"
    print(f"  ✅ {tag:<28} 校验通过 ({suffix})")


N, C, BLOCK_N, eps = 4096, 4096, 1, 1e-5
x = torch.randn(N, C, dtype=torch.float16, device="cuda")
w = torch.randn(C, dtype=torch.float16, device="cuda")

# ─────────────── 1) 整行版 (基准) ───────────────
k_full = tl_rmsnorm_forward.compile(N=N, C=C, BLOCK_N=BLOCK_N, eps=eps)
y_full, m2_full = k_full(x, w)
check(y_full, m2_full, x, w, eps, "整行版")

# ─────────────── 2) 分块版 (扫几个 BLOCK_C) ───────────────
BLOCK_C_LIST = (256, 512, 1024)
kernels = {}
for BLOCK_C in BLOCK_C_LIST:
    assert C % BLOCK_C == 0, f"BLOCK_C={BLOCK_C} 不能整除 C={C}"
    k = tl_rmsnorm_forward_splitc.compile(
        N=N, C=C, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C, eps=eps
    )
    y, m2 = k(x, w)

    check(y, m2, x, w, eps, f"分块版 BLOCK_C={BLOCK_C}")
    # 两个版本互相对照 (比只跟 ref 比更严格: 能发现"两边一起错"的共模问题)
    torch.testing.assert_close(y, y_full, rtol=1e-2, atol=1e-2)
    kernels[BLOCK_C] = k

# ─────────────── 3) 测速 ───────────────
print()
t_full = do_bench(lambda: k_full(x, w), warmup=25, rep=100)
print(f"  整行版 (BLOCK_C = C)      : {t_full:7.3f} ms")

for BLOCK_C in BLOCK_C_LIST:
    t = do_bench(lambda: kernels[BLOCK_C](x, w), warmup=25, rep=100)
    print(f"  分块版 BLOCK_C={BLOCK_C:<5}        : {t:7.3f} ms   ({t_full / t:5.2f}x vs 整行版)")

t_ref = do_bench(lambda: ref_rmsnorm(x, w, eps), warmup=25, rep=100)
print(f"  torch 参考实现            : {t_ref:7.3f} ms   (TileLang 整行版 {t_ref / t_full:5.2f}x)")

print("\n✅ 全部通过")

