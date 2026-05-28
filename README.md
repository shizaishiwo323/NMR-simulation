# NMR Simulation Package

这个文件夹给新用户使用时，根目录只需要先看三个文件：

```text
Auto_NMR.py
simulation_control.py
NMR_Computation_Logic.tex
```

最简单的运行方式：

```powershell
python Auto_NMR.py
```

`Auto_NMR.py` 是主程序，`simulation_control.py` 是控制文件。一般只改控制文件，不需要改主程序。

`NMR_Computation_Logic.tex` 是底层计算逻辑说明，适合第一次上手时从 PDE、pyGIMLi、T2/T2-T2/D-T2 反演开始理解。

## 选择输出哪些内容

打开 `simulation_control.py`，修改：

```python
"modules": {
    "uncoupled_t2": True,
    "coupled_t2": True,
    "t2_t2": True,
    "dt2": True,
}
```

含义：

- `uncoupled_t2`: 未耦合 T2
- `coupled_t2`: 耦合 T2
- `t2_t2`: T2-T2 exchange map
- `dt2`: D-T2 / PFG map

只想快速算 T2：

```python
"modules": {
    "uncoupled_t2": True,
    "coupled_t2": True,
    "t2_t2": False,
    "dt2": False,
}
```

只想算未耦合：

```python
"modules": {
    "uncoupled_t2": True,
    "coupled_t2": False,
    "t2_t2": False,
    "dt2": False,
}
```

## 选择 fixed alpha 或 L-curve

打开 `simulation_control.py`，修改：

```python
"inversion": {
    "mode": "fixed",
    "alpha": 1.0,
    "image_mode": "l_curve",
}
```

- 理想三角孔或无噪声模拟：通常用 `"fixed"`。
- 玻璃珠/CT 分割图像：建议用 `"l_curve"`。

## 交互模式

如果想让程序运行时询问参数：

```python
"run": {
    "interactive": True,
}
```

如果想稳定复现实验，建议保持：

```python
"interactive": False
```

## 输出位置

默认输出到：

```text
simulation_outputs/Auto_NMR_controlled/
```

运行后会生成：

- `NMR_T2_Figures/`
- `T2_Inversion_Detailed_2.5D.xlsx`
- `Triangle_Raw_Decay.xlsx`
- `T2_T2_Exchange_Map.xlsx`
- `DT2_Uncoupled_Map.xlsx`
- `DT2_Coupled_Map.xlsx`
- `Simulation_Control_Used.json`

## 文件夹说明

```text
nmr_t2/              L-curve、NNLS、Gaussian、pipeline 反演包
simulation_suite/    模块化辅助代码
data/input/          示例 CT/玻璃珠切片 Result.tif
examples/            已经算好的一组示例结果
advanced_tools/      高级脚本，需要时再看
```

## 高级脚本

不懂代码时不用看 `advanced_tools/`。

里面包含：

- `Auto_T1.py`: T1 模拟
- `image_slice_simulation.py`: 玻璃珠/CT 分割图输入
- `png_phase_nmr_decay.py`: 红/黄/白 PNG phase map 输入，输出 NMR T2 弛豫曲线
- `run_triangle_contact_angle_cases.py`: 多接触角/三角形批量计算
- `triangle_contact_angle_full_suite.py`: 单组三角孔完整 T2/T2-T2/D-T2
- `CA_Drain_and_imbi.py`: 接触角排水/吸水水分分布
- `make_combined_figures.py`: 组合图
- `organize_triangle_outputs.py`: 整理输出目录

如需运行高级脚本，请在包根目录运行，并设置 `PYTHONPATH` 指向当前目录：

```powershell
$env:PYTHONPATH='.'
python advanced_tools/image_slice_simulation.py
```

红色代表水相、黄色代表固体、白色代表外部区域的 PNG 可以直接运行：

```powershell
$env:PYTHONPATH='.'
python advanced_tools/png_phase_nmr_decay.py "C:\path\to\interface_images" --output-dir simulation_outputs/png_phase_nmr_decay --pattern "timestep_*.png"
```

如果知道图像对应的实际物理尺寸，建议直接传入横向和纵向长度：

```powershell
$env:PYTHONPATH='.'
python advanced_tools/png_phase_nmr_decay.py "C:\path\to\interface_images\timestep_0002.png" --output-dir simulation_outputs/png_phase_nmr_decay --length-x-cm 0.0575 --length-y-cm 0.042
```

默认边界解释为：内部红-黄接触是固液边界，左/右白边接触是气液边界，上/下白边接触是固液边界。`--rho-solid-um-per-ms`、`--rho-gas-um-per-ms`、`--diffusion-um2-per-ms`、`--bulk-t2-ms`、`--length-x-cm` 和 `--length-y-cm` 应按实验或文献参数确认后设置。

如果需要三角网格而不是像素有限差分网格，可以启用：

```powershell
$env:PYTHONPATH='.'
python advanced_tools/png_phase_nmr_decay.py "C:\path\to\interface_images\timestep_0001.png" --output-dir simulation_outputs/png_tri_mesh --length-x-cm 0.0575 --length-y-cm 0.037176914536239794 --solver triangular --mesh-bulk-size-um 10 --mesh-boundary-size-um 2.5
```

`--mesh-bulk-size-um` 控制水相内部点间距，数值越小整体越细；`--mesh-boundary-size-um` 控制水-固/水-气边界采样间距，建议小于 bulk size，使靠近边界的网格更密。三角网格会同时输出预览图、pyGIMLi 可读取的 `.bms` 网格文件、逐单元质量 CSV 和网格质量直方图。

## 示例

看 `EXAMPLES.md`。

本包已包含：

- `data/input/Result.tif`
- `examples/image_slice_Result/`
- `examples/triangle_CA0deg_angles60-60-60/`
