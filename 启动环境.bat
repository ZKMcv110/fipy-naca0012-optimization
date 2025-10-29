
@echo off
echo 正在激活NACA0012翼型扰流柱优化项目环境...
echo.

REM 激活虚拟环境
call myenvs_fipynaca2.0\Scripts\activate

echo 环境已激活!
echo.
echo 可用命令:
echo  ========================
echo  python 网格生成器2_三角形.py     - 生成网格
echo  python compute_Nu_f.py           - 运行CFD计算
echo  python generate_samples.py       - 生成参数样本
echo  python run_all_cases.py          - 批量运行案例
echo  python train_ai_models.py        - 训练AI模型
echo  python ai_optimization.py        - AI优化
echo  ========================
echo.
echo 输入命令开始工作，或直接按回车键进入Python环境...


cmd /k