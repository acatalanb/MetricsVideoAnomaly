@echo off
set DATASET_NAME=crime-ucf
set EPOCHS=20

echo 🚀 Starting Training and Evaluation for all models on %DATASET_NAME% dataset...

:: 1. CNN-LSTM
echo --------------------------------------------------
echo 🟦 Training CNN-LSTM...
python train.py --model "CNN-LSTM" --epochs %EPOCHS% --dataset_name %DATASET_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo ❌ CNN-LSTM Training failed!
) else (
    echo ✅ CNN-LSTM Training complete.
    echo 🔍 Evaluating CNN-LSTM...
    python evaluate.py --model "CNN-LSTM" --model_path "cache/model_CNN_LSTM.pth" --dataset_name %DATASET_NAME% --epochs %EPOCHS%
)

:: 2. 3D CNN
echo --------------------------------------------------
echo 🟥 Training 3D CNN...
python train.py --model "3D CNN" --epochs %EPOCHS% --dataset_name %DATASET_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 3D CNN Training failed!
) else (
    echo ✅ 3D CNN Training complete.
    echo 🔍 Evaluating 3D CNN...
    python evaluate.py --model "3D CNN" --model_path "cache/model_3D_CNN.pth" --dataset_name %DATASET_NAME% --epochs %EPOCHS%
)

:: 3. Video Transformer
echo --------------------------------------------------
echo 🟩 Training Video Transformer...
python train.py --model "Video Transformer" --epochs %EPOCHS% --dataset_name %DATASET_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Video Transformer Training failed!
) else (
    echo ✅ Video Transformer Training complete.
    echo 🔍 Evaluating Video Transformer...
    python evaluate.py --model "Video Transformer" --model_path "cache/model_Video_Transformer.pth" --dataset_name %DATASET_NAME% --epochs %EPOCHS%
)

echo --------------------------------------------------
echo 📊 Generating Comparison Results...
python compare_models.py --dataset_name %DATASET_NAME%
echo ✅ All tasks complete!
pause
