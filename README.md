# Module 5: ML Debugging and Diagnostics (PyTorch)

Практическая работа по отладке и диагностике ML-систем.

## Что сделано
1. Контрольный запуск: Обучение ForecastLSTM на синусоиде. Модель (MSE 0.0038) обходит baseline (MSE 0.0048).
2. Отладчик pdb: Проверка размерностей тензоров через breakpoint().
3. Диагностика ошибок:
   - --fault no-step: Симуляция забытого optimizer.step(). В логах видно, что delta_weight = 0.0, модель не обновляется.
   - --fault bad-labels: Перемешивание таргетов. Модель не может найти закономерность, loss не падает.
4. TensorBoard: Логирование Loss/train, Loss/test, Diagnostics/grad_norm и delta_weight.

## Запуск
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy tensorboard matplotlib psutil

python train.py
python train.py --debug --epochs 2
python train.py --fault no-step --epochs 5
tensorboard --logdir runs --port 6006
