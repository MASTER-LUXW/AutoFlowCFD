"""AutoFlowCFD V2.0 - des 的自测/演示入口。

从 `src/autoflowcfd/core/turbulence/des.py` 的 `if __name__ == "__main__"` 块搬来(2026-09-24 拆包)。
跑法从 `python src/autoflowcfd/core/turbulence/des.py` 变成 `python -m autoflowcfd.core.turbulence.des`。
"""

import numpy as np


from . import DDESModel


if __name__ == "__main__":
    # 测试代码
    from autoflowcfd.core.turbulence.sst import SSTModelFR
    
    # 创建测试数据
    n_cells = 100
    n_sps = 8
    
    d_w = np.random.rand(n_cells, n_sps) * 0.01
    cell_volumes = np.ones(n_cells) * 1e-6
    nu_t = np.random.rand(n_cells, n_sps) * 1e-4
    omega = np.random.rand(n_cells, n_sps) * 100
    
    # 创建 SST 模型
    sst = SSTModelFR(n_cells, n_sps)
    sst.k_field = np.random.rand(n_cells, n_sps) * 1e-4
    
    # 应用 DDES
    ddes = DDESModel()
    ddes.apply_to_sst_model(sst, d_w, cell_volumes, nu=np.full((n_cells, n_sps), 1.5e-5))
    
    print("DDES model test completed.")
