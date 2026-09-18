"""
rmsnorm backward kernel apply by tilelang
"""

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import torch

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


# ════════════════════════════════════════════════════════════════
#  版本 1: 整行版 —— 一个 fragment 装下整行 C 个元素
#  (C 很大时 BLOCK_N 只能取 1~4, 寄存器装不下)
# ════════════════════════════════════════════════════════════════
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
        sum_local= T.alloc_fragment((BLOCK_N,), accum_dtype)
        dX_local = T.alloc_fragment((BLOCK_N, C), io_dtype)
        dW_part  = T.alloc_fragment((BLOCK_N, C), accum_dtype)   # 块内 dW 的原料
        dW_chunk = T.alloc_fragment((C,), accum_dtype)           # 块内 dW 的结果

        T.copy(dY[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :], dY_local)
        T.copy(X[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :], X_local)
        T.copy(W, W_shared)
        T.copy(mean2[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N], rstd)

        # inv
        for i in T.Parallel(BLOCK_N):
            rstd[i] = T.rsqrt(rstd[i] + eps)

        for i, j in T.Parallel(BLOCK_N, C):
            tmp3[i, j] = dY_local[i, j].astype(accum_dtype) * X_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype)
        T.reduce_sum(tmp3, sum_local, dim=1, clear=True)

        for i in T.Parallel(BLOCK_N):
            sum_local[i] = sum_local[i] * rstd[i] * rstd[i] / C

        for i, j in T.Parallel(BLOCK_N, C):
            dX_local[i, j] = dY_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype) * rstd[i] - X_local[i, j].astype(accum_dtype) * rstd[i] * sum_local[i]

            dW_part[i, j] = dY_local[i, j].astype(accum_dtype) * X_local[i, j].astype(accum_dtype) * rstd[i]
        T.reduce_sum(dW_part, dW_chunk, dim=0, clear=True)

        T.copy(dX_local, dX[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :])
        for j in T.Parallel(C):
            T.atomic_add(dW[j], dW_chunk[j], memory_order="relaxed")

    return dX


# ════════════════════════════════════════════════════════════════
#  版本 2: splitc 分块版 —— 手心只开 (BLOCK_N, BLOCK_C), C 方向用 T.Serial 分块
#  这样 BLOCK_N 不再被整行 C 卡死, 而是由"面积" BLOCK_N × BLOCK_C 决定
# ════════════════════════════════════════════════════════════════
@tilelang.jit
def tl_rmsnorm_backward_splitc(dY, X, W, mean2, dW, BLOCK_N: int, BLOCK_C: int, eps: float):
    N, C = T.const("N, C")
    assert C % BLOCK_C == 0, f"BLOCK_C={BLOCK_C} 不能整除 C={C}"
    io_dtype = T.float16
    accum_dtype = T.float32
    dY: T.Tensor((N, C), io_dtype)
    X: T.Tensor((N, C), io_dtype)
    W: T.Tensor((C,), io_dtype)
    mean2: T.Tensor((N,), accum_dtype)
    dX = T.empty((N, C), io_dtype)
    dW: T.Tensor((C,), accum_dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as pid_n:
        dY_local = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        X_local  = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        W_shared = T.alloc_shared((BLOCK_C,), io_dtype)
        rstd     = T.alloc_fragment((BLOCK_N,), accum_dtype)
        tmp3     = T.alloc_fragment((BLOCK_N, BLOCK_C), accum_dtype)
        sum_local= T.alloc_fragment((BLOCK_N,), accum_dtype)
        dX_local = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        dW_part  = T.alloc_fragment((BLOCK_N, BLOCK_C), accum_dtype)
        dW_chunk = T.alloc_fragment((BLOCK_C,), accum_dtype)

        T.copy(mean2[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N], rstd)

        # inv
        for i in T.Parallel(BLOCK_N):
            rstd[i] = T.rsqrt(rstd[i] + eps)

        num_c_step = T.ceildiv(C, BLOCK_C)

        # ── 第一趟: 跨步累加 Σ_j dy·x·w ──
        T.clear(tmp3)                       # ★ 累积器开工前必须清零
        for k in T.Serial(num_c_step):
            T.copy(dY[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, k * BLOCK_C : (k + 1) * BLOCK_C], dY_local)
            T.copy(X[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, k * BLOCK_C : (k + 1) * BLOCK_C], X_local)
            T.copy(W[k * BLOCK_C : (k + 1) * BLOCK_C], W_shared)

            for i, j in T.Parallel(BLOCK_N, BLOCK_C):
                tmp3[i, j] += dY_local[i, j].astype(accum_dtype) * X_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype)
        T.reduce_sum(tmp3, sum_local, dim=1, clear=True)

        for i in T.Parallel(BLOCK_N):
            sum_local[i] = sum_local[i] * rstd[i] * rstd[i] / C

        # ── 第二趟: 倒序, 蹭 L2 cache (第一趟最后读的那段此刻还热着) ──
        for k in T.Serial(num_c_step):
            kk = num_c_step - 1 - k
            T.copy(dY[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, kk * BLOCK_C : (kk + 1) * BLOCK_C], dY_local)
            T.copy(X[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, kk * BLOCK_C : (kk + 1) * BLOCK_C], X_local)
            T.copy(W[kk * BLOCK_C : (kk + 1) * BLOCK_C], W_shared)

            for i, j in T.Parallel(BLOCK_N, BLOCK_C):
                dX_local[i, j] = dY_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype) * rstd[i] - X_local[i, j].astype(accum_dtype) * rstd[i] * sum_local[i]

                dW_part[i, j] = dY_local[i, j].astype(accum_dtype) * X_local[i, j].astype(accum_dtype) * rstd[i]
            T.reduce_sum(dW_part, dW_chunk, dim=0, clear=True)

            T.copy(dX_local, dX[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, kk * BLOCK_C : (kk + 1) * BLOCK_C])
            for j in T.Parallel(BLOCK_C):
                T.atomic_add(dW[kk * BLOCK_C + j], dW_chunk[j], memory_order="relaxed")

    return dX


# ═══════════════════════════════════════════════════════════════
#  测试: RMSNorm backward  (整行版 vs splitc 分块版)
# ═══════════════════════════════════════════════════════════════


def ref_rmsnorm_bwd(x, w, dy, eps=1e-5):
    """参考实现: 用 torch autograd 求 dX / dW

    为什么不用手写公式当参考: 手写公式万一推错, 两边会一起错, 测试就白测了。
    autograd 是完全独立的第三方裁判。(官方 examples/norm/layernorm.py 也是这么对拍的)

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


def check(dX, dW, x, w, dy, eps, tag=""):
    """校验 dX 和 dW —— kernel 有两个输出, 两项都必须查"""
    ref_dX, ref_dW = ref_rmsnorm_bwd(x, w, dy, eps)

    torch.testing.assert_close(dX.float(), ref_dX, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dW.float(), ref_dW, rtol=1e-2, atol=1e-2)

    err_x = (dX.float() - ref_dX).abs().max().item()
    err_w = (dW.float() - ref_dW).abs().max().item()
    print(f"  ✅ {tag:<26} 校验通过 (dX + dW, max|ΔdX|={err_x:.1e}, max|ΔdW|={err_w:.1e})")


N, C, eps = 1024, 4096, 1e-5
torch.manual_seed(0)          # 梯度测试要可复现, 否则报错没法重查

x  = torch.randn(N, C, dtype=torch.float16, device="cuda")
w  = torch.randn(C,    dtype=torch.float16, device="cuda")
dy = torch.randn(N, C, dtype=torch.float16, device="cuda")

# 前向保存给反向的中间量 (fp32): mean2 = mean(x^2)
mean2 = x.float().pow(2).mean(dim=-1)

# ─────────────── 1) 整行版 (基准, 扫几个 BLOCK_N) ───────────────
BLOCK_N_LIST = (1, 32, 128)
kernels = {}
dX_base = None
dW_base = None

for BN in BLOCK_N_LIST:
    assert N % BN == 0, f"BLOCK_N={BN} 不能整除 N={N}"
    k = tl_rmsnorm_backward.compile(N=N, C=C, BLOCK_N=BN, eps=eps)

    # ★ dW 是 atomic_add 的账本: 必须由调用方清零, 而且必须是 fp32
    dW = torch.zeros(C, dtype=torch.float32, device="cuda")
    dX = k(dy, x, w, mean2, dW)

    check(dX, dW, x, w, dy, eps, f"整行版 BLOCK_N={BN}")

    # 不同分块方式互相对照: 结果必须一致
    if dX_base is None:
        dX_base, dW_base = dX, dW.clone()
    else:
        torch.testing.assert_close(dX, dX_base, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(dW, dW_base, rtol=1e-2, atol=1e-2)

    kernels[BN] = k

# ─────────────── 2) splitc 分块版: 面积不变, 形状变 ───────────────
# (BLOCK_N, BLOCK_C) 的【乘积】就是"每组手心要装多少格"。
# 前 5 个组合乘积都是 4096 —— 手心占用完全一样, 只是切成了不同形状,
# 所以它们的耗时差异纯粹来自"分块形状", 这才是我们要看的东西。
SPLITC_COMBOS = (
    (1,  4096),   # 面积 4096   ← 等价于整行版
    (2,  2048),   # 面积 4096
    (4,  1024),   # 面积 4096
    (8,   512),   # 面积 4096
    (16,  256),   # 面积 4096
    (8,  1024),   # 面积 8192   ← 超寄存器预算, 用来看"装不下"的代价
)
split_kernels = {}

for BN, BC in SPLITC_COMBOS:
    assert N % BN == 0, f"BLOCK_N={BN} 不能整除 N={N}"
    assert C % BC == 0, f"BLOCK_C={BC} 不能整除 C={C}"
    k = tl_rmsnorm_backward_splitc.compile(N=N, C=C, BLOCK_N=BN, BLOCK_C=BC, eps=eps)

    dW = torch.zeros(C, dtype=torch.float32, device="cuda")
    dX = k(dy, x, w, mean2, dW)

    check(dX, dW, x, w, dy, eps, f"splitc BN={BN} BC={BC}")

    # 跟整行版对照 (两个独立实现, 必须给出同一个答案)
    torch.testing.assert_close(dX, dX_base, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dW, dW_base, rtol=1e-2, atol=1e-2)

    split_kernels[(BN, BC)] = k

# ─────────────── 3) 测速 ───────────────
print()


def bench(k):
    dW = torch.zeros(C, dtype=torch.float32, device="cuda")

    def run():
        dW.zero_()                       # 每轮前清零, 保证每次跑的条件一致
        k(dy, x, w, mean2, dW)

    return do_bench(run, warmup=25, rep=100)


t_base = bench(kernels[BLOCK_N_LIST[0]])
print(f"  整行版 BLOCK_N={BLOCK_N_LIST[0]:<5}          : {t_base:7.3f} ms   (基准)")
for BN in BLOCK_N_LIST[1:]:
    t = bench(kernels[BN])
    print(f"  整行版 BLOCK_N={BN:<5}          : {t:7.3f} ms   ({t / t_base:5.2f}x)")

print()
for BN, BC in SPLITC_COMBOS:
    t = bench(split_kernels[(BN, BC)])
    print(f"  splitc BN={BN:<3} BC={BC:<5} (面积 {BN * BC:<5}): {t:7.3f} ms   ({t / t_base:5.2f}x)")

# torch autograd 参考耗时 (只能粗略参考: autograd 还有图调度/元数据开销)
xr = x.detach().clone().requires_grad_(True)
wr = w.detach().clone().requires_grad_(True)
xrf = xr.float()
yr = wr.float() * xrf * torch.rsqrt(xrf.pow(2).mean(dim=-1, keepdim=True) + eps)


def ref_run():
    xr.grad = None
    wr.grad = None
    yr.backward(dy.float(), retain_graph=True)


t_ref = do_bench(ref_run, warmup=25, rep=100)
print()
print(f"  torch autograd 参考实现             : {t_ref:7.3f} ms   (TileLang 最快 {t_ref / t_base:5.2f}x)")

print("\n✅ 全部通过")


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
# check(dX, dW, x, w, dy, eps, "端到端 (用前向的 mean2)")
