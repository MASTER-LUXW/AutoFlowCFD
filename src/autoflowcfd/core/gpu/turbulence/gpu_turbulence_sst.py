"""AutoFlowCFD V2.0 - GPU 上的 SST k-omega 模型：`SSTModelFR` 的设备版本。

公式（源项、混合函数、涡粘、正性/上界、点隐式更新、DES 长度尺度）与 CPU 是同一份数组模块无关的
实现（`core/turbulence/sst/`），本类只把场数组放到指定的 GPU 上。此前这里另有一份逐行复制的
cupy 实现，CPU 侧的修复多次只落在一份上（F1 溢出保护、末尾 isfinite 归零、粘性比上限都曾只在
CPU 上）。

使用:
    from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
    gpu_sst = GPUTurbulenceSST(n_cells, n_sps, device_id=0)
    Sk, S_omega = gpu_sst.compute_source_terms(Q_gpu, grad_U_gpu, d_wall_gpu, mu, grad_k_gpu, grad_omega_gpu)
"""

from autoflowcfd.core.gpu import gpu_available, get_cupy
from autoflowcfd.core.turbulence.sst.model import SSTModelFR


class GPUTurbulenceSST(SSTModelFR):
    """场数组常驻 GPU 的 SST k-omega 模型。

    Attributes:
        device_id: GPU 设备 ID（其余属性同 `SSTModelFR`）
    """

    def _new_instance(self, n_cells: int, n_sps: int):
        return type(self)(n_cells, n_sps, self.device_id, k_inf=self.k_inf, omega_inf=self.omega_inf)

    def __init__(self, n_cells: int, n_sps: int, device_id: int = 0,
                 k_inf: float = 1e-6, omega_inf: float = 1.0):
        """
        Args:
            n_cells: 单元数量
            n_sps: 每单元解点数
            device_id: GPU 设备 ID
            k_inf / omega_inf: 来流湍动能 / 比耗散率（由 Tu/VR 推导，见 `fr_solver/turbulence/init.py`）
        """
        if not gpu_available:
            raise RuntimeError("CuPy required for GPU turbulence model")
        cp = get_cupy()
        self.device_id = device_id
        with cp.cuda.Device(device_id):
            super().__init__(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf, xp=cp)
