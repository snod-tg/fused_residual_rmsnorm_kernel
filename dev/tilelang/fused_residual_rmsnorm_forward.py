"""
fused residual rmsnorm forward kernel by tilelang

融合了什么: 把 [残差相加 z = x + residual] 塞进 RMSNorm 前向, 省掉一次显存往返。
另外把 z 写出去存给反向 —— 反向就不用再重算 x + residual 了。

对照 dev/fused_residual_rmsnorm_forward.cu 的 CPU 参考:

    for(int row = 0; row < N; ++row){
        for(int c = 0; c < C; ++c){
            zr[c] = xr[c] + residualr[c];     // ★ 融合进来的残差相加
            sum += zr[c] * zr[c];
        }
        float m2 = sum / C;
        mean2[row] = m2;
        float inv = 1.0f / std::sqrt(m2 + kEps);
        for(int c = 0; c < C; ++c) yr[c] = zr[c] * w[c] * inv;
    }

三个输入: x, residual, w
三个输出: y, z, mean2
"""

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import torch


# ════════════════════════════════════════════════════════════════
#  版本 1: 整行版 —— 一个 fragment 装下整行 C 个元素
#  z 只读一遍: x/r 搬进手心 → 算出 z 留在寄存器 → 算完直接写 y
# ════════════════════════════════════════════════════════════════
@tilelang.jit
def tl_fused_rmsnorm_forward(X, R, W, BLOCK_N: int, eps: float):
    N, C = T.const("N, C")
    io_dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((N, C), io_dtype)
    R: T.Tensor((N, C), io_dtype)
    W: T.Tensor((C,), io_dtype)
    Y = T.empty((N, C), io_dtype)
    Z = T.empty((N, C), io_dtype)
    mean2 = T.empty((N,), accum_dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as pid_n:
        X_local   = T.alloc_fragment((BLOCK_N, C), io_dtype)
        R_local   = T.alloc_fragment((BLOCK_N, C), io_dtype)
        Z_local   = T.alloc_fragment((BLOCK_N, C), io_dtype)     # z 既是中间量, 也是输出
        Z2_local  = T.alloc_fragment((BLOCK_N, C), accum_dtype)
        Y_local   = T.alloc_fragment((BLOCK_N, C), io_dtype)
        W_shared  = T.alloc_shared((C,), io_dtype)               # 广播数据放小桌子
        mean2_local = T.alloc_fragment((BLOCK_N,), accum_dtype)
        tmp_sum   = T.alloc_fragment((BLOCK_N,), accum_dtype)

        T.copy(X[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :], X_local)
        T.copy(R[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :], R_local)
        T.copy(W, W_shared)

        # ★ 融合点: x + residual = z, 顺便算 z²
        for i, j in T.Parallel(BLOCK_N, C):
            Z_local[i, j] = X_local[i, j].astype(accum_dtype) + R_local[i, j].astype(accum_dtype)
            Z2_local[i, j] = Z_local[i, j].astype(accum_dtype) * Z_local[i, j].astype(accum_dtype)

        T.reduce_sum(Z2_local, tmp_sum, dim=1, clear=True)
        for i in T.Parallel(BLOCK_N):
            mean2_local[i] = tmp_sum[i] / C

        for i, j in T.Parallel(BLOCK_N, C):
            Y_local[i, j] = Z_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype) * T.rsqrt(mean2_local[i] + eps)

        T.copy(Z_local, Z[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :])       # z 存给反向
        T.copy(Y_local, Y[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, :])
        T.copy(mean2_local, mean2[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N])

    return Y, Z, mean2


# ════════════════════════════════════════════════════════════════
#  版本 2: splitc 分块版 —— 手心只开 (BLOCK_N, BLOCK_C), C 方向 T.Serial 分块
#  注意第二趟读的是【z】而不是 x/r —— 第一趟已经把 z 算出来写出去了,
#  直接读一份 z 比重读 x+r 两份省一半带宽。
# ════════════════════════════════════════════════════════════════
@tilelang.jit
def tl_fused_rmsnorm_forward_splitc(X, R, W, BLOCK_N: int, BLOCK_C: int, eps: float):
    N, C = T.const("N, C")
    # assert C % BLOCK_C == 0, f"BLOCK_C={BLOCK_C} 不能整除 C={C}"
    io_dtype = T.float16
    accum_dtype = T.float32
    X: T.Tensor((N, C), io_dtype)
    R: T.Tensor((N, C), io_dtype)
    W: T.Tensor((C,), io_dtype)
    Y = T.empty((N, C), io_dtype)
    Z = T.empty((N, C), io_dtype)
    mean2 = T.empty((N,), accum_dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as pid_n:
        Z_local   = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        X_local   = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        R_local   = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        Z2_local  = T.alloc_fragment((BLOCK_N, BLOCK_C), accum_dtype)
        Y_local   = T.alloc_fragment((BLOCK_N, BLOCK_C), io_dtype)
        W_shared  = T.alloc_shared((BLOCK_C,), io_dtype)
        mean2_local = T.alloc_fragment((BLOCK_N,), accum_dtype)
        tmp_sum   = T.alloc_fragment((BLOCK_N,), accum_dtype)

        num_c_step = T.ceildiv(C, BLOCK_C)

        # ── 第一趟: 算 z 并写出, 同时跨步累加 Σz² ──
        T.clear(Z2_local)                     # 累积器开工前必须清零
        for k in T.Serial(num_c_step):
            T.copy(X[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, k * BLOCK_C : (k + 1) * BLOCK_C], X_local)
            T.copy(R[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, k * BLOCK_C : (k + 1) * BLOCK_C], R_local)

            for i, j in T.Parallel(BLOCK_N, BLOCK_C):
                Z_local[i, j] = X_local[i, j].astype(accum_dtype) + R_local[i, j].astype(accum_dtype)
                Z2_local[i, j] += Z_local[i, j].astype(accum_dtype) * Z_local[i, j].astype(accum_dtype)

            T.copy(Z_local, Z[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, k * BLOCK_C : (k + 1) * BLOCK_C])

        T.reduce_sum(Z2_local, tmp_sum, dim=1, clear=True)
        for i in T.Parallel(BLOCK_N):
            mean2_local[i] = tmp_sum[i] / C

        # ── 第二趟: 只读 z (一份), 倒序蹭 L2 cache ──
        for k in T.Serial(num_c_step):
            kk = num_c_step - 1 - k
            T.copy(Z[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, kk * BLOCK_C : (kk + 1) * BLOCK_C], Z_local)
            T.copy(W[kk * BLOCK_C : (kk + 1) * BLOCK_C], W_shared)

            for i, j in T.Parallel(BLOCK_N, BLOCK_C):
                Y_local[i, j] = Z_local[i, j].astype(accum_dtype) * W_shared[j].astype(accum_dtype) * T.rsqrt(mean2_local[i] + eps)

            T.copy(Y_local, Y[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N, kk * BLOCK_C : (kk + 1) * BLOCK_C])

        T.copy(mean2_local, mean2[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N])

    return Y, Z, mean2


# ═══════════════════════════════════════════════════════════════
#  测试: 融合残差 RMSNorm forward  (整行版 vs splitc 分块版)
# ═══════════════════════════════════════════════════════════════


def ref_fused_rmsnorm_fwd(x, r, w, eps=1e-5):
    """参考实现: z = x + r; mean2 = mean(z^2); y = z * w * rsqrt(mean2 + eps)

    z 刻意先 round 成 fp16 再往下算 —— 跟 kernel 内部一致
    (kernel 里 Z_local 是 fp16, 后面 z² 和 y 都基于这个已经舍入过的 z)
    """
    z = (x.float() + r.float()).to(x.dtype)
    zf = z.float()
    mean2 = zf.pow(2).mean(dim=-1)
    y = (zf * w.float() * torch.rsqrt(mean2.unsqueeze(-1) + eps)).to(x.dtype)
    return y, z, mean2


def check(y, z, mean2, x, r, w, eps, tag=""):
    """三个输出都要查"""
    ref_y, ref_z, ref_mean2 = ref_fused_rmsnorm_fwd(x, r, w, eps)

    torch.testing.assert_close(z, ref_z, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(mean2, ref_mean2, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(y, ref_y, rtol=1e-2, atol=1e-2)

    print(f"  ✅ {tag:<26} 校验通过 (y + z + mean2)")


N, C, BLOCK_N, eps = 4096, 4096, 1, 1e-5
torch.manual_seed(0)
x = torch.randn(N, C, dtype=torch.float16, device="cuda")
r = torch.randn(N, C, dtype=torch.float16, device="cuda")
w = torch.randn(C, dtype=torch.float16, device="cuda")

# ─────────────── 1) 整行版 (基准) ───────────────
k_full = tl_fused_rmsnorm_forward.compile(N=N, C=C, BLOCK_N=BLOCK_N, eps=eps)
y_full, z_full, m2_full = k_full(x, r, w)
check(y_full, z_full, m2_full, x, r, w, eps, "整行版")

# ─────────────── 2) splitc 分块版: 二维扫描 ───────────────
# (BLOCK_N, BLOCK_C) 的【乘积】= 每组手心要装多少格。
# 前 5 个组合面积都是 4096 —— 占用一样, 形状不同。
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
    k = tl_fused_rmsnorm_forward_splitc.compile(N=N, C=C, BLOCK_N=BN, BLOCK_C=BC, eps=eps)

    y, z, m2 = k(x, r, w)
    check(y, z, m2, x, r, w, eps, f"splitc BN={BN} BC={BC}")

    # 跟整行版对照 (两个独立实现, 必须给出同一个答案)
    torch.testing.assert_close(y, y_full, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(z, z_full, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(m2, m2_full, rtol=1e-2, atol=1e-2)

    split_kernels[(BN, BC)] = k

# ─────────────── 3) 测速 ───────────────
print()
t_full = do_bench(lambda: k_full(x, r, w), warmup=25, rep=100)
print(f"  整行版 (C 方向不切块)            : {t_full:7.3f} ms")

times = {}
for BN, BC in SPLITC_COMBOS:
    t = do_bench(lambda: split_kernels[(BN, BC)](x, r, w), warmup=25, rep=100)
    times[(BN, BC)] = t
    print(f"  splitc BN={BN:<3} BC={BC:<5} (面积 {BN * BC:<5}): {t:7.3f} ms   ({t / t_full:5.2f}x vs 整行版)")

best = min(times, key=times.get)
t_best = times[best]
print(f"\n  最快: splitc BN={best[0]} BC={best[1]} = {t_best:.3f} ms  (比整行版快 {t_full / t_best:.2f}x)")

t_ref = do_bench(lambda: ref_fused_rmsnorm_fwd(x, r, w, eps), warmup=25, rep=100)
print(f"  torch 参考实现                   : {t_ref:7.3f} ms   (TileLang 最快 {t_ref / t_best:5.2f}x)")

print("\n✅ 全部通过")
