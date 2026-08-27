"""对比 100/200/300 步的流场状态，定位发散模式。"""
import h5py
import numpy as np

BASE = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\steady_results\checkpoints"
nu = 1.8e-5 / 1.225
k_max = 0.5 * 33.33 ** 2

for it in [100, 200, 300]:
    ckpt = f"{BASE}\\checkpoint_iter_{it:06d}.h5"
    with h5py.File(ckpt, 'r') as f:
        print(f"\n{'='*20} iter {it} {'='*20}")
        # 残差历史
        try:
            res = f['convergence/history/residuals'][:]
            print(f"  residual history (last 10): {res[-10:]}")
            print(f"  residual (first 5): {res[:5]}")
        except Exception as e:
            print(f"  no residual history: {e}")

        U = f['solution/U_sps'][:]
        rho = U[:, 0, 0]
        u = U[:, 0, 1] / np.maximum(rho, 1e-10)
        v = U[:, 0, 2] / np.maximum(rho, 1e-10)
        w = U[:, 0, 3] / np.maximum(rho, 1e-10)
        p = U[:, 0, 4]
        speed = np.sqrt(u**2 + v**2 + w**2)
        print(f"  rho: mean={rho.mean():.4f} min={rho.min():.4f} max={rho.max():.4f}")
        print(f"  |V|: mean={speed.mean():.2f} min={speed.min():.2f} max={speed.max():.2f}")
        print(f"  u:   mean={u.mean():.2f} max={u.max():.2f}")
        print(f"  p:   mean={p.mean():.0f} min={p.min():.0f} max={p.max():.0f}")
        print(f"  u<0 占比(回流): {(u<0).mean()*100:.1f}%")

        k = f['solution/k_field'][:, 0]
        om = f['solution/omega_field'][:, 0]
        nt = f['solution/nu_t'][:, 0]
        print(f"  k:     mean={k.mean():.3e} median={np.median(k):.3e} max={k.max():.3e}")
        print(f"  omega: mean={om.mean():.3e} median={np.median(om):.3e} max={om.max():.3e}")
        print(f"  nu_t:  max={nt.max():.3e} (max ratio={nt.max()/nu:.1f})")
        print(f"  k@下界(1e-12) 占比: {(k <= 1.01e-12).mean()*100:.1f}%")
        print(f"  k@上界 占比: {(k >= 0.99*k_max).mean()*100:.1f}%")
        print(f"  k 中间值占比: {((k > 1.01e-12) & (k < 0.99*k_max)).mean()*100:.1f}%")
        print(f"  omega@下界 占比: {(om <= 1.01e-12).mean()*100:.1f}%")
        print(f"  omega@上界(1e6) 占比: {(om >= 0.99e6).mean()*100:.1f}%")
        # 诊断: k/omega 比值 -> nu_t 无限制时的值
        ratio = k / np.maximum(om, 1e-10)
        print(f"  k/omega: median={np.median(ratio):.3e} max={ratio.max():.3e}")

print("\nDone.")
