# 如何上传这份代码和数据

这份目录就是仓库内容。先解压收到的 ZIP，再把目录里面的文件和子目录放到现有 `SNIa-Evolution-CDDR` 仓库根目录。不要只上传 ZIP，否则网页上只能看到压缩包，运行说明和代码也不便查看。

## 上传哪些文件

本目录的代码、数据、参考结果、文档和校验清单都要上传，包括根目录 README、LICENSE、requirements、`reproduce.py`、`verify_package.py`、`.gitignore`，以及 `code/`、`data/`、`baseline/`、`original/`、`reference_results/`、`docs/`。

不要把运行后产生的 `_runs/`、解压生成的 33 MB `.cov`、本地日志、其他论文工程或私人审稿资料加进来；`.gitignore` 已列出运行产物。

`data/Pantheon+SH0ES_STAT+SYS.cov.gz` 是完整原协方差的无损压缩，约 10 MB，运行时会自动恢复并核对原哈希。这里没有删掉非对角项，也没有降低数字精度。

## 更新已有仓库

1. 保留已有 Git 历史，先为初稿版本建立标签或备份。
2. 把本目录内容复制到仓库工作副本根目录，保留本包的目录层次。
3. 旧根目录里的 `CosmoMatcher_v1.1.py`、`mcmc_sn_evolutionv5m.py`、`cosmo_tools.py` 及旧配对文件已在本包 `original/` 中归档；更新时避免让旧入口与新入口在根目录并列造成误用。
4. 在科学环境运行 `python verify_package.py`，再运行 `python reproduce.py --check`。
5. 用 GitHub Desktop 或 Git 提交并推送到你的仓库。使用网页上传时，可分批上传各个文件夹；保持 `.gitignore` 等根文件也在。
6. 打开仓库网页，核对能看到新的 README、`reproduce.py`、`code/gp.py`、`code/bounded_inference.py`、`code/scatter_sensitivity.py`、三个输入文件及参考结果目录。

GitHub 网页上传的单文件上限是 25 MiB。本包每个应上传文件均低于此限制；外层总 ZIP 可能大于该限制，所以应先解压再上传目录。[GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

## 上传后再更新论文

核对公开仓库确实包含此次版本后，论文才能准确写新版代码已公开。若创建带版本号的 Release 或 DOI 存档，应引用那个真实版本，不能把初稿旧 DOI 当成这份修订包的 DOI。本次准备没有替你操作 GitHub、创建 DOI 或填写未经确认的软件贡献者名单。

根 MIT 许可证沿用你现有分析仓库的原文本；数据和 bundled CosmoMatcher 的权利范围单独列在 `THIRD_PARTY.md`，没有擅自替上游增加许可。

## 本包已经包含的功能

- 原始数据和对应完整协方差。
- 原版 48 对贪心配对及四模型复现代码。
- 保留原先验、加入固定 10% 角尺寸误差的基准。
- 真正两阶段 MILP、严格红移窗、有符号运输及背景敏感性。
- 全协方差 GP、48/50 同锚点比较、核/均值与五折检查。
- 线性/对数函数组合、先验边界、经验额外散布。
- 四幅基准三角图和三幅诊断图的生成入口。
- 全部 42 组补充后验、参考结果、文件哈希和运行核对。
