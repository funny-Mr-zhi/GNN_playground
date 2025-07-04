@echo off
REM 图神经网络学习环境设置脚本 (Windows版本)

echo 🚀 设置图神经网络学习环境...

REM 创建conda环境
echo 📦 创建conda环境...
call conda create -n gnn-learning python=3.9 -y
call conda activate gnn-learning

REM 安装PyTorch（根据系统和CUDA版本调整）
echo 🔥 安装PyTorch...
REM CPU版本
REM call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

REM CUDA 11.8版本
call conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

REM 安装PyTorch Geometric
echo 📐 安装PyTorch Geometric...
call pip install torch-geometric
call pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

REM 安装DGL
echo ⚡ 安装DGL...
call pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

REM 安装其他依赖
echo 📚 安装其他依赖...
call pip install -r requirements.txt

REM 安装Jupyter扩展
echo 📓 设置Jupyter环境...
call conda install -c conda-forge nodejs -y
call jupyter labextension install @jupyter-widgets/jupyterlab-manager
call jupyter labextension install plotlywidget

REM 创建Jupyter kernel
call python -m ipykernel install --user --name gnn-learning --display-name "GNN Learning"

REM 验证安装
echo ✅ 验证安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyTorch Geometric版本: {torch_geometric.__version__}')"
python -c "import dgl; print(f'DGL版本: {dgl.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

echo 🎉 环境设置完成！
echo 使用以下命令激活环境：
echo conda activate gnn-learning

pause
