#!/bin/bash

# 安装必要的工具
pip install --upgrade pip build twine

# 清理之前的构建
rm -rf dist/ build/ *.egg-info/

# 构建包
python -m build

echo "打包完成！"
echo "分发包在 dist/ 目录下。"
echo ""
echo "测试上传到 TestPyPI:"
echo "twine upload --repository testpypi dist/*"
echo ""
echo "正式上传到 PyPI:"
echo "twine upload dist/*" 