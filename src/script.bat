@echo off
REM Script to compile, run experiments, and generate visualizations
REM Big Data Analytics Programming - Assignment 1
REM Spam Filter Experiments (Windows version)

echo ==========================================
echo Spam Filter Experiment Pipeline
echo ==========================================
echo.

REM Step 1: Compile the project
echo [1/3] Compiling C++ code...
echo -------------------------------------------

REM Create build directory if it doesn't exist
if not exist "..\build" (
    mkdir ..\build
)

cd ..\build

REM Run CMake
echo Running CMake...
cmake ..

if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed!
    exit /b 1
)

REM Compile
echo Compiling...
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo ERROR: Compilation failed!
    exit /b 1
)

echo [✓] Compilation successful
echo.

REM Step 2: Run the experiments
echo [2/3] Running experiments...
echo -------------------------------------------
echo This will take several minutes...
echo.

REM Run with default seed=12 and window=200
src\Release\bdap_assignment1.exe 12 200

if %errorlevel% neq 0 (
    echo ERROR: Experiment execution failed!
    exit /b 1
)

echo.
echo [✓] Experiments complete
echo.

REM Step 3: Generate visualizations
echo [3/3] Generating visualizations...
echo -------------------------------------------

cd ..\src

REM Run visualization script
python plot.py

if %errorlevel% neq 0 (
    echo ERROR: Visualization generation failed!
    echo Note: Make sure Python and required packages are installed:
    echo   pip install pandas matplotlib seaborn numpy
    exit /b 1
)

echo.
echo [✓] Visualizations generated
echo.

REM Summary
echo ==========================================
echo Pipeline Complete!
echo ==========================================
echo.
echo Generated files:
echo   CSV Results:
echo     - learning_curves.csv
echo     - threshold_experiments.csv
echo     - filter_experiments.csv
echo     - hyperparameter_results.csv
echo     - timing_results.csv
echo     - best_configurations.csv
echo.
echo   Visualizations:
echo     - learning_curves.png
echo     - learning_curves_focused.png
echo     - threshold_comparison.png
echo     - bucket_filter_impact.png
echo     - hyperparameter_heatmap_fh.png
echo     - hyperparameter_heatmap_cm.png
echo     - precision_recall_tradeoff.png
echo     - computational_efficiency.png
echo.
echo All experiments completed successfully!

pause

