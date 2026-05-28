# Advanced Tools

这些脚本不是新手主入口。默认只运行根目录的：

```text
Auto_NMR.py
```

如果确实需要运行本文件夹里的高级脚本，请从包根目录执行，并设置：

```powershell
$env:PYTHONPATH='.'
```

示例：

```powershell
$env:PYTHONPATH='.'
python advanced_tools/image_slice_simulation.py
python advanced_tools/run_triangle_contact_angle_cases.py --contact-angles 0 30 45 --triangle-angles 60 60 60
```

脚本用途：

- `Auto_T1.py`: T1 模拟
- `image_slice_simulation.py`: 玻璃珠/CT 分割图像输入
- `run_triangle_contact_angle_cases.py`: 多接触角和三角形角度批量模拟
- `triangle_contact_angle_full_suite.py`: 单组三角孔完整 T2/T2-T2/D-T2
- `CA_Drain_and_imbi.py`: 接触角排水/吸水水分分布
- `make_combined_figures.py`: 组合图
- `organize_triangle_outputs.py`: 整理输出结果
