"""
rmsnorm backward kernel apply by tilelang
"""

import tilelang
import tilelang.language as T

"""
void rmsnorm_backward_cpu(
    float* dx,
    float* dw,
    const float* dy,
    const float* x,
    const float* weight,
    const float* mean2,
    int N,
    int C
){
    std::fill(dw, dw + C, 0.0f);
    for(int row = 0; row < N; ++row){
        const float* xr = x + (size_t)row * C;
        const float* dyr = dy + (size_t)row * C;
        float* dxr = dx + (size_t)row * C;

        float sum = 0.0f;
        for(int c = 0; c < C; ++c){
            sum += dyr[c] * weight[c] * xr[c];
        }

        float inv = 1.0f / std::sqrt(mean2[row] + kEps);
        float correction = sum * inv * inv / C;

        for(int c = 0; c < C; ++c){
            dxr[c] = dyr[c] * weight[c] * inv - xr[c] * inv * correction;
            dw[c] += dyr[c] * xr[c] * inv;
        }
    }
}
"""

@tilelang.jit
def tl_rmsnorm_backward(dY, X, W, mean2, dW, BLOCK_N: int, eps: float):
    N, C = T.const("N, C")
    io_dtype = T.float16
    accum_dtype = T.float32
    dY: T.Tensor((N, C), io_dtype)
    X: T.Tensor((N, C), io_dtype)
    W: T.Tensor((C,), io_dtype)
    mean2: T.Tensor((N,), accum_dtype)
    dX = T.empty((N, C), io_dtype)
    dW: T.Tensor((C,), accum_dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as pid_n:
        dY_local = T.alloc_fragment((BLOCK_N, C), io_dtype)
        X_local  = T.alloc_fragment((BLOCK_N, C), io_dtype)
        W_shared = T.alloc_shared((C,), io_dtype)          # ← 广播数据放小桌子
        rstd     = T.alloc_fragment((BLOCK_N,), accum_dtype)
        tmp3     = T.alloc_fragment((BLOCK_N, C), accum_dtype)
        sum_local= T.alloc_fragment((BLOCK_N, C), accum_dtype)
        corr     = T.alloc_fragment((BLOCK_N,), accum_dtype)
        dX_local = T.alloc_fragment((BLOCK_N, C), io_dtype)
        dW_part  = T.alloc_fragment((BLOCK_N, C), accum_dtype)   # 块内 dW 的原料
        dW_chunk = T.alloc_fragment((C,), accum_dtype)           # 块内 dW 的结果

        T.copy(dY[pid_n * BLOCK_N, :], dY_local)
        T.copy(X[pid_n * BLOCK_N, :], X_local)
        T.copy(W, W_shared)
        T.copy(mean2[pid_n * BLOCK_N], rstd)

        # inv
        for i in T.Parallel(BLOCK_N):
            rstd[i] = T.rsqrt(rstd[i] + eps)

        for i, j in T.Parallel(BLOCK_N, C):
            tmp3[i, j] = dY_local[i, j].astype(accum_dtype) * X_local[i, j].astype(accum_dtype)  * W_shared[j].astype(accum_dtype) 
        T.reduce_sum(tmp3, sum_local, dim=1, clear=True)

        for i in T.Parallel(BLOCK_N):
            sum_local[i] = sum_local[i] * rstd[i] * rstd[i] / C

        for i, j in T.Parallel(BLOCK_N, C):
            dX_local[i, j] = dY_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype) * rstd[i] - X_local[i, j].astype(accum_dtype) * rstd[i] * sum_local[i]

            dW_part[i, j] = dY_local[i, j].astype(accum_dtype) * X_local[i, j].astype(accum_dtype) * rstd[i]
        T.reduce_sum(dW_part, dW_chunk, dim=0, clear=True)


        T.copy(dX_local, dX[pid_n * BLOCK_N, :])
        for j in T.Parallel(C):
            T.atomic_add(dW[j], dW_chunk[j], memory_order="relaxed")

    return dX


# ════════════════════════════════════════════════════════════════
#  测试: RMSNorm backward  ——  跟 torch autograd 对拍
#
#  参考实现用 autograd, 而不是自己手写公式:
#  手写公式万一推错, 两边会一起错; autograd 是独立的第三方裁判。
#  (官方 examples/norm/layernorm.py 就是这么对拍的)
# ════════════════════════════════════════════════════════════════
import torch
from tilelang.profiler import do_bench


def torch_rmsnorm_bwd(x, w, dy, eps=1e-5):
    """参考实现: 用 autograd 求 dX / dW

    前向刻意写在 fp32 里 —— 跟 kernel 内部一致
    (kernel 里除了最后写回是 fp16, 计算全程 fp32)
    """
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)

    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = w.float() * xf * rstd
    y.backward(dy.float())

    return x.grad, w.grad


def check_bwd(dX, dW, x, w, dy, eps, tag=""):
    """逐项对拍: kernel 有几个输出, 测试就查几项 (dX 和 dW 都必须查)"""
    dX_ref, dW_ref = torch_rmsnorm_bwd(x, w, dy, eps)

    torch.testing.assert_close(dX.float(), dX_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dW.float(), dW_ref, rtol=1e-2, atol=1e-2)

    err_x = (dX.float() - dX_ref).abs().max().item()
    err_w = (dW.float() - dW_ref).abs().max().item()
    print(f"  ✅ {tag:<20} dX ✅  dW ✅   (max|ΔdX|={err_x:.2e}, max|ΔdW|={err_w:.2e})")


N, C, eps = 1024, 4096, 1e-5
torch.manual_seed(0)

x = torch.randn(N, C, dtype=torch.float16, device="cuda")
w = torch.randn(C, dtype=torch.float16, device="cuda")
dy = torch.randn(N, C, dtype=torch.float16, device="cuda")

# 前向要保存给反向的中间量 (fp32): mean2 = mean(x^2)
mean2 = x.float().pow(2).mean(dim=-1)

# ── 扫几个 BLOCK_N: 1 = "一行一组", 32/128 = "多行一组" ──
for BLOCK_N in (1, 32, 128):
    kernel = tl_rmsnorm_backward.compile(N=N, C=C, BLOCK_N=BLOCK_N, eps=eps)

    # ★ dW 是 atomic_add 的账本: 必须由调用方清零, 而且必须是 fp32
    dW = torch.zeros(C, dtype=torch.float32, device="cuda")
    dX = kernel(dy, x, w, mean2, dW)

    check_bwd(dX, dW, x, w, dy, eps, f"BLOCK_N={BLOCK_N}")

    # ── 测速 (每轮前清零, 保证每次跑的条件一致) ──
    def run():
        dW.zero_()
        kernel(dy, x, w, mean2, dW)

    t = do_bench(run, warmup=25, rep=100)
    print(f"      TileLang: {t:.3f} ms")

print("\n✅ RMSNorm backward 全部通过")


# ────────────────────────────────────────────────────────────────
#  进阶: 用前向 kernel 真正存下来的 mean2 来跑反向 (端到端更真实)
#  前向的 mean2 输出本来就是 fp32, 可以直接喂给反向
# ────────────────────────────────────────────────────────────────
# from rmsnorm_forward import tl_rmsnorm_forward
#
# k_fwd = tl_rmsnorm_forward.compile(N=N, C=C, BLOCK_N=1, eps=eps)
# y, mean2_fwd = k_fwd(x, w)
#
# dW = torch.zeros(C, dtype=torch.float32, device="cuda")
# k_bwd = tl_rmsnorm_backward.compile(N=N, C=C, BLOCK_N=1, eps=eps)
# dX = k_bwd(dy, x, w, mean2_fwd, dW)
# check_bwd(dX, dW, x, w, dy, eps, "端到端 (用前向的 mean2)")

        

