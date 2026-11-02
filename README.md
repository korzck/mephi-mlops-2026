# Credit Default Prediction

Пайплайн для PD-модели (вероятность дефолта по кредитной карте), построенный
на датасете Default of Credit Card Clients с UCI Machine Learning Repository.

## Структура проекта

```
src/
  data/           загрузка данных и валидация (Great Expectations)
  features/       feature engineering
  models/         sklearn pipeline, обучение, метрики
  api/            FastAPI приложение
tests/            unit-тесты
scripts/          вспомогательные скрипты (Docker, имитация дрифта)
data/raw          исходные данные (версионируются DVC)
data/processed    train/test выборки
models/           обученная модель и ROC-кривая
```

## Установка

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Подготовка данных и обучение

Через DVC пайплайн (две стадии: prepare и train):

```
dvc repro
```

Или вручную:

```
python -m src.data.make_dataset
python -m src.models.train
```

Обучение логирует эксперименты в MLflow. Посмотреть результаты:

```
mlflow ui --backend-store-uri file:./mlruns
```

## Валидация данных

```
python -m src.data.validation data/processed/train.csv
```

## Тесты и линтинг

```
pytest tests/ -q
flake8 src tests
black --check src tests
```

## API

Локально:

```
uvicorn src.api.app:app --reload
```

Через Docker:

```
bash scripts/build_and_run_docker.sh
```

Пример запроса:

```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
  "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
  "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
  "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
}'
```

## Мониторинг дрифта

Скрипт шлет часть тестовой выборки на запущенный API и считает PSI
относительно train распределения:

```
python -m scripts.simulate_drift
```
