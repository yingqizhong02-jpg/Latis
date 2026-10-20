# 第三方组件与版权说明

本仓库的代码（MIT）之外，程序运行还依赖若干第三方库，以及若干名称/商标。
本文件记录它们各自的来源与许可。

---

## 一、运行依赖

| 组件 | 用途 | 许可 |
|---|---|---|
| [rawpy](https://github.com/letmaik/rawpy) | RAW 解码的 Python 绑定 | MIT |
| [LibRaw](https://www.libraw.org/) | RAW 解码底层库（经 rawpy） | LGPL-2.1 / CDDL-1.0（双许可） |
| [NumPy](https://numpy.org/) | 数值运算 | BSD-3-Clause |
| [Pillow](https://python-pillow.org/) | 图像读写与合成 | MIT-CMU |
| [customtkinter](https://github.com/TomSchimansky/CustomTkinter) | 界面控件 | MIT |
| [tkinter / Tcl / Tk](https://www.tcl.tk/) | 窗口系统（Python 标准库） | Tcl/Tk License（BSD 风格） |
| [PyInstaller](https://pyinstaller.org/) | 打包为 exe | GPL-2.0 **带例外条款**（允许打包专有程序） |

发布包（`Latis-1.0.zip`）内含上述库的二进制分发。各自的完整许可文本随
PyInstaller 的 `_internal` 目录与其官方分发一并提供；如需单独获取，请见各项目主页。

> LibRaw 采用 LGPL-2.1 / CDDL-1.0 双许可。本项目**以动态链接方式使用其未修改的二进制**，
> 未修改 LibRaw 源码。如你重新分发本程序，请遵守 LGPL 关于二进制分发的要求
> （提供许可文本、允许替换该库）。

---

## 二、LUT 文件

**本仓库不包含任何 LUT 文件。**

`lut/` 目录只保留结构说明，使用者需自行提供 `.cube` 文件。

早期版本（v1.0 初版发布包）曾包含一批来源与许可不明的 LUT，
现已**从仓库、提交历史与发布包中全部移除**。

如果你在本仓库的旧版本中见到任何 `.cube` 文件，请**不要**使用或再分发。

---

## 三、示例中出现的名称

程序内置的色彩空间名称（Rec.709、Rec.2020、sRGB、Display P3、ACES、
ARRI LogC3、Sony S-Log3、Panasonic V-Log、Fujifilm F-Log2、DaVinci Intermediate 等）
以及它们对应的色域/传递函数定义，来自各标准的**公开技术规范**与
[ACES](https://www.oscars.org/science-technology/aces)（Academy of Motion Picture Arts and Sciences）公开文档。

相机与胶片品牌相关的名称——**FUJIFILM、Fujifilm Film Simulation、Classic Chrome、
Classic Negative、Nostalgic Negative、ACROS、ASTIA、PROVIA、Velvia、ETERNA、
Kodak、Vision3、ARRI、Sony、Panasonic / LUMIX、Leica、RED、Adobe、Photoshop、
Lightroom、DaVinci Resolve / Blackmagic** ——均为其各自权利人的商标。

本程序使用这些名称**仅用于说明**：
- 相机兼容性（能读哪种 RAW）
- 软件兼容性（能与哪种 LUT 格式配合）
- 目标色彩风格或传递函数的**技术名称**

**本项目与上述任何公司均无官方关联、授权、合作或背书关系。**

---

## 四、免责

不同的 RAW 文件、镜头、曝光、白平衡、软件版本与显示设备，可能产生不同结果。

程序保证的是：文件中的像素值与色彩空间标记自洽。
程序不保证你的屏幕发光标准（这需要色度计验证）。

建议保留原始 RAW 与相机原厂软件，作为参照。

使用者自行决定是否安装和使用本程序。完整免责条款见 [免责声明.txt](免责声明.txt)。

---

*如果你认为本仓库中有内容侵犯了你的权利，请开 issue 说明，我会立即处理。*
