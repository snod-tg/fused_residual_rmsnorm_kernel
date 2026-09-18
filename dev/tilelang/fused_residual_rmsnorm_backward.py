"""
fused residual rmsnorm backward kernel by tilelang

对应 dev/fused_residual_rmsnorm_backward.cu 的 CPU 参考:

    for(int row = 0; row < N; ++row){
        float sum = 0.0f;
        for(int c = 0; c < C; ++c) sum += dyr[c] * weight[c] * zr[c];

        float inv = 1.0f / std::sqrt(mean2[row] + kEps);
        float correction = sum * inv * inv / C;

        for(int c = 0; c < C; ++c){
            float v = dyr[c] * weight[c] * inv - zr[c] * inv * correction;
            dxr[c] = v;
            dresidualr[c] = v;          // ★ dx 和 dresidual 完全一样!
            dw[c] += dyr[c] * zr[c] * inv;
        }
    }

两个要点:
1. 反向用的是前向存下来的【z】, 不是 x/residual —— 这正是融合的意义:
   前向已经把 z 算出来了, 反向不必再重算一次 x + r。
2. 因为 z = x + residual, ∂z/∂x = ∂z/∂residual = 1,
   所以 dx 和 dresidual 是同一个值 —— 这里让它们【共用同一块 fragment】, 省一整个 (BLOCK_N, C)。

四个输入: dy, z, weight, mean2
三个输出: dx, dresidual, dw
"""

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import torch


# ════════════════════════════════════════════════════════════════
#  版本 1: 整行版 —— 一个 fragment 装下整行 C 个元素
# ════════════════════════════════════════════════════════════════
@tilelang.jit
def tl_fused_rmsnorm_backward(dY, Z, W, mean2, dW, BLOCK_N: int, eps: float):
    N, C = T.const("N, C")
    io_dtype = T.float16
    accum_dtype = T.float32
    dY: T.Tensor((N, C), io_dtype)
    Z: T.Tensor((N, C), io_dtype)
    W: T.Tensor((C,), io_dtype)
    mean2: T.Tensor((N,), accum_dtype)
    dW: T.Tensor((C,), accum_dtype)          # ★ atomic_add 的账本, 由调用方清零
    dX = T.empty((N, C), io_dtype)
    dR = T.empty((N, C), io_dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as pid_n:
        dY_local = T.alloc_fragment((BLOCK_N, C), io_dtype)
        Z_local  = T.alloc_fragment((BLOCK_N, C), io_dtype)
        W_shared = T.alloc_shared((C,), io_dtype)
        rstd     = T.alloc_fragment((BLOCK_N,), accum_dtype)
        tmp3     = T.alloc_fragment((BLOCK_N, C), accum_dtype)
        sum_local= T.alloc_fragment((BLOCK_N,), accum_dtype)
        dX_local = T.alloc_fragment((BLOCK_N, C), io_dtype)      # dx / dresidual 共用
        dW_part  = T.alloc_fragment((BLOCK_N, C), accum_dtype)
        dW_chunk = T.alloc_fragment((C,), accum_dtype)

        T.copy(dY[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :], dY_local)
        T.copy(Z[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :], Z_local)
        T.copy(W, W_shared)
        T.copy(mean2[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N], rstd)

        for i in T.Parallel(BLOCK_N):
            rstd[i] = T.rsqrt(rstd[i] + eps)          # inv

        for i, j in T.Parallel(BLOCK_N, C):
            tmp3[i, j] = dY_local[i, j].astype(accum_dtype) * Z_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype)
        T.reduce_sum(tmp3, sum_local, dim=1, clear=True)

        for i in T.Parallel(BLOCK_N):
            sum_local[i] = sum_local[i] * rstd[i] * rstd[i] / C     # correction

        for i, j in T.Parallel(BLOCK_N, C):
            dX_local[i, j] = (dY_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype) * rstd[i]
                              - Z_local[i, j].astype(accum_dtype) * rstd[i] * sum_local[i])
            dW_part[i, j] = dY_local[i, j].astype(accum_dtype) * Z_local[i, j].astype(accum_dtype) * rstd[i]
        T.reduce_sum(dW_part, dW_chunk, dim=0, clear=True)

        T.copy(dX_local, dX[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :])
        T.copy(dX_local, dR[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :])   # ★ dresidual == dx
        for j in T.Parallel(C):
            T.atomic_add(dW[j], dW_chunk[j], memory_order="relaxed")

    return dX, dR


# ════════════════════════════════════════════════════════════════
#  版本 2: splitc 分块版 —— 手心只开 (BLOCK_N, BLOCK_C)
# ════════════════════════════════════════════════════════════════
@tilelang.jit
def tl_fused_rmsnorm_backward_splitc(dY, Z, W, mean2, dW, BLOCK_N: int, BLOCK_C: int, eps: float):
    N, C = T.const("N, C")
    # assert C % BLOCK_C == 0, f"BLOCK_C={BLOCK_C} 不能整除 C={C}"
    io_dtype = T.float16
    accum_dtype = T.float32
    dY: T.Tensor((N, C), io_dtype)
    Z: T.Tensor((N, C), io_dtype)
    W: T.Tensor((C,), io_dtype)
    mean2: T.Tensor((N,), accum_dtype)
    dW: T.Tensor((C,), accum_dtype)
    dX = T.empty((N, C), io_dtype)
    dR = T.empty((N, C), io_dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as pid_n:
        dY_local = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        Z_local  = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        W_shared = T.alloc_shared((BLOCK_C,), io_dtype)
        rstd     = T.alloc_fragment((BLOCK_N,), accum_dtype)
        tmp3     = T.alloc_fragment((BLOCK_N, BLOCK_C), accum_dtype)
        sum_local= T.alloc_fragment((BLOCK_N,), accum_dtype)
        dX_local = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)     # dx / dresidual 共用
        dW_part  = T.alloc_fragment((BLOCK_N, BLOCK_C), accum_dtype)
        dW_chunk = T.alloc_fragment((BLOCK_C,), accum_dtype)

        T.copy(mean2[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N], rstd)
        for i in T.Parallel(BLOCK_N):
            rstd[i] = T.rsqrt(rstd[i] + eps)

        num_c_step = T.ceildiv(C, BLOCK_C)

        # ── 第一趟: 跨步累加 Σ_j dy·z·w ──
        T.clear(tmp3)                        # 累积器开工前必须清零
        for k in T.Serial(num_c_step):
            T.copy(dY[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, k * BLOCK_C : (k + 1) * BLOCK_C], dY_local)
            T.copy(Z[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, k * BLOCK_C : (k + 1) * BLOCK_C], Z_local)
            T.copy(W[k * BLOCK_C : (k + 1) * BLOCK_C], W_shared)

            for i, j in T.Parallel(BLOCK_N, BLOCK_C):
                tmp3[i, j] += dY_local[i, j].astype(accum_dtype) * Z_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype)

        T.reduce_sum(tmp3, sum_local, dim=1, clear=True)
        for i in T.Parallel(BLOCK_N):
            sum_local[i] = sum_local[i] * rstd[i] * rstd[i] / C

        # ── 第二趟: 倒序, 蹭 L2 cache ──
        for k in T.Serial(num_c_step):
            kk = num_c_step - 1 - k
            T.copy(dY[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, kk * BLOCK_C : (kk + 1) * BLOCK_C], dY_local)
            T.copy(Z[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, kk * BLOCK_C : (kk + 1) * BLOCK_C], Z_local)
            T.copy(W[kk * BLOCK_C : (kk + 1) * BLOCK_C], W_shared)

            for i, j in T.Parallel(BLOCK_N, BLOCK_C):
                dX_local[i, j] = (dY_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype) * rstd[i]
                                  - Z_local[i, j].astype(accum_dtype) * rstd[i] * sum_local[i])
                dW_part[i, j] = dY_local[i, j].astype(accum_dtype) * Z_local[i, j].astype(accum_dtype) * rstd[i]

            T.reduce_sum(dW_part, dW_chunk, dim=0, clear=True)

            T.copy(dX_local, dX[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, kk * BLOCK_C : (kk + 1) * BLOCK_C])
            T.copy(dX_local, dR[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, kk * BLOCK_C : (kk + 1) * BLOCK_C])
            for j in T.Parallel(BLOCK_C):
                T.atomic_add(dW[kk * BLOCK_C + j], dW_chunk[j], memory_order="relaxed")

    return dX, dR


# ═══════════════════════════════════════════════════════════════
#  测试: 融合残差 RMSNorm backward  (整行版 vs splitc 分块版)
# ═══════════════════════════════════════════════════════════════


def ref_fused_rmsnorm_bwd(x, r, w, dy, eps=1e-5):
    """参考实现: 用 torch autograd (独立第三方裁判, 不会跟 kernel 一起错)

    前向: z = x + r; y = z * w * rsqrt(mean(z^2) + eps)
    反向: 求 x.grad / r.grad / w.grad
    """
    x = x.detach().clone().requires_grad_(True)
    r = r.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)

    zf = x.float() + r.float()
    mean2 = zf.pow(2).mean(dim=-1, keepdim=True)
    y = zf * w.float() * torch.rsqrt(mean2 + eps)
    y.backward(dy.float())

    return x.grad, r.grad, w.grad


def check(dX, dR, dW, x, r, w, dy, eps, tag=""):
    """三个输出都要查; 另外验证融合语义: dx 必须等于 dresidual"""
    ref_dX, ref_dR, ref_dW = ref_fused_rmsnorm_bwd(x, r, w, dy, eps)

    # ★ 融合语义检查: z = x + r  =>  dx == dresidual
    torch.testing.assert_close(dX, dR, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dX.float(), ref_dX.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dR.float(), ref_dR.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dW.float(), ref_dW.float(), rtol=1e-2, atol=1e-2)

    print(f"  ✅ {tag:<26} 校验通过 (dx + dresidual + dw, 且 dx == dresidual)")


N, C, eps = 1024, 4096, 1e-5
torch.manual_seed(0)

x  = torch.randn(N, C, dtype=torch.float16, device="cuda")
r  = torch.randn(N, C, dtype=torch.float16, device="cuda")
w  = torch.randn(C,    dtype=torch.float16, device="cuda")
dy = torch.randn(N, C, dtype=torch.float16, device="cuda")

# 前向存给反向的中间量: z 和 mean2
z = (x.float() + r.float()).to(torch.float16)
mean2 = z.float().pow(2).mean(dim=-1)


def new_dW():
    """dW 是 atomic_add 的账本: 必须由调用方清零, 而且必须是 fp32"""
    return torch.zeros(C, dtype=torch.float32, device="cuda")


# ─────────────── 1) 整行版 (基准, 扫几个 BLOCK_N) ───────────────
BLOCK_N_LIST = (1, 32, 128)
kernels = {}
dX_base = dR_base = None
dW_base = None

for BN in BLOCK_N_LIST:
    assert N % BN == 0, f"BLOCK_N={BN} 不能整除 N={N}"
    k = tl_fused_rmsnorm_backward.compile(N=N, C=C, BLOCK_N=BN, eps=eps)

    dW = new_dW()
    dX, dR = k(dy, z, w, mean2, dW)

    check(dX, dR, dW, x, r, w, dy, eps, f"整行版 BLOCK_N={BN}")

    if dX_base is None:
        dX_base, dR_base, dW_base = dX, dR, dW.clone()
    else:
        torch.testing.assert_close(dX, dX_base, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(dW, dW_base, rtol=1e-2, atol=1e-2)

    kernels[BN] = k

# ─────────────── 2) splitc 分块版: 二维扫描 ───────────────
SPLITC_COMBOS = (
    (1,  4096),   # 面积 4096   ← 等价于整行版
    (2,  2048),   # 面积 4096
    (4,  1024),   # 面积 4096
    (8,   512),   # 面积 4096
    (16,  256),   # 面积 4096
    (4,  2048),   # 面积 8192
    (8,  1024),   # 面积 8192
)
split_kernels = {}

for BN, BC in SPLITC_COMBOS:
    assert N % BN == 0, f"BLOCK_N={BN} 不能整除 N={N}"
    assert C % BC == 0, f"BLOCK_C={BC} 不能整除 C={C}"
    k = tl_fused_rmsnorm_backward_splitc.compile(N=N, C=C, BLOCK_N=BN, BLOCK_C=BC, eps=eps)

    dW = new_dW()
    dX, dR = k(dy, z, w, mean2, dW)

    check(dX, dR, dW, x, r, w, dy, eps, f"splitc BN={BN} BC={BC}")

    # 跟整行版对照 (两个独立实现, 必须给出同一个答案)
    torch.testing.assert_close(dX, dX_base, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dW, dW_base, rtol=1e-2, atol=1e-2)

    split_kernels[(BN, BC)] = k

# ─────────────── 3) 测速 ───────────────
print()


def bench(k):
    dW = new_dW()

    def run():
        dW.zero_()
        k(dy, z, w, mean2, dW)

    return do_bench(run, warmup=25, rep=100)


t_base = bench(kernels[BLOCK_N_LIST[0]])
print(f"  整行版 BLOCK_N={BLOCK_N_LIST[0]:<5}          : {t_base:7.3f} ms   (基准)")
for BN in BLOCK_N_LIST[1:]:
    t = bench(kernels[BN])
    print(f"  整行版 BLOCK_N={BN:<5}          : {t:7.3f} ms   ({t / t_base:5.2f}x)")

print()
times = {}
for BN, BC in SPLITC_COMBOS:
    t = bench(split_kernels[(BN, BC)])
    times[(BN, BC)] = t
    print(f"  splitc BN={BN:<3} BC={BC:<5} (面积 {BN * BC:<5}): {t:7.3f} ms   ({t / t_base:5.2f}x)")

best = min(times, key=times.get)
t_best = times[best]
print(f"\n  最快: splitc BN={best[0]} BC={best[1]} = {t_best:.3f} ms  (比整行版基准快 {t_base / t_best:.2f}x)")

# torch autograd 参考耗时 (只能粗略参考: autograd 还有图调度/元数据开销)
xr = x.detach().clone().requires_grad_(True)
rr = r.detach().clone().requires_grad_(True)
wr = w.detach().clone().requires_grad_(True)
zrf = xr.float() + rr.float()
m2r = zrf.pow(2).mean(dim=-1, keepdim=True)
yr = zrf * wr.float() * torch.rsqrt(m2r + eps)


def ref_run():
    xr.grad = None
    rr.grad = None
    wr.grad = None
    yr.backward(dy.float(), retain_graph=True)


t_ref = do_bench(ref_run, warmup=25, rep=100)
print(f"  torch autograd 参考实现             : {t_ref:7.3f} ms   (TileLang 最快 {t_ref / t_best:5.2f}x)")

print("\n✅ 全部通过")
