@echo off
echo ================================
echo NACA0012翼型扰流柱优化流程
echo ================================
echo.

REM 检查是否在项目目录
if not exist "网格生成器2_三角形.py" (
    echo 错误: 请在项目根目录运行此脚本
    pause
    exit /b 1
)

REM 激活虚拟环境
echo 正在激活虚拟环境...
call myenvs_fipynaca2.0\Scripts\activate
echo 虚拟环境已激活
echo.

REM 询问用户是否开始完整流程
echo 本流程将执行以下步骤:
echo 1. 生成参数样本 (250个案例)
echo 2. 批量运行CFD计算 (可能需要数小时)
echo 3. 训练AI模型
echo 4. 进行AI优化
echo 5. 输出最优参数
echo.
set /p confirm=是否开始完整优化流程? (y/N): 

if /i not "%confirm%"=="y" (
    echo 流程已取消
    pause
    exit /b 0
)

echo.
echo ================================
echo 步骤1: 生成参数样本
echo ================================
python generate_samples.py
if %errorlevel% neq 0 (
    echo 错误: 参数样本生成失败
    pause
    exit /b %errorlevel%
)

echo.
echo ================================
echo 步骤2: 批量运行CFD计算
echo ================================
echo 此步骤可能需要较长时间，请耐心等待...
python run_all_cases.py
if %errorlevel% neq 0 (
    echo 错误: CFD计算失败
    pause
    exit /b %errorlevel%
)

echo.
echo ================================
echo 步骤3: 训练AI模型
echo ================================
python train_ai_models.py
if %errorlevel% neq 0 (
    echo 错误: AI模型训练失败
    pause
    exit /b %errorlevel%
)

echo.
echo ================================
echo 步骤4: 进行AI优化
echo ================================
python ai_optimization.py
if %errorlevel% neq 0 (
    echo 错误: AI优化失败
    pause
    exit /b %errorlevel%
)

echo.
echo ================================
echo 优化完成!
echo ================================
echo 请查看 optimal_parameters.txt 文件获取最优参数
echo.
echo 如需验证结果，请使用以下命令运行一次验证:
echo python run_optimization_case.py ^<Tt^> ^<Ts^> ^<Tad^> ^<Tb^>
echo.
pause