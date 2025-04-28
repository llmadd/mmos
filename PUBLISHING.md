# 发布指南

本文档介绍如何将MMOS包发布到PyPI。

## 前提条件

确保已安装以下工具：

```bash
pip install build twine
```

## 构建包

在项目根目录下运行：

```bash
python -m build
```

这将在`dist/`目录下创建源码分发包和wheel包。

## 测试上传

在正式发布前，建议先上传到TestPyPI进行测试：

```bash
python -m twine upload --repository testpypi dist/*
```

然后可以从TestPyPI安装进行测试：

```bash
pip install --index-url https://test.pypi.org/simple/ mmos
```

## 正式发布

确认测试无误后，上传到PyPI：

```bash
python -m twine upload dist/*
```

## 更新版本

更新版本时，修改`setup.py`和`mmos/__init__.py`中的版本号，然后重复上述步骤。

## PyPI账户注册

如果还没有PyPI账户，需要在 https://pypi.org/account/register/ 注册一个账户。 