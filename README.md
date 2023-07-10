###   Обучение модели детекции и классификации переломов на медицинских изображениях

### 

#### Установка conda окружения 
Для создания и активации conda окружения необходимо выполнить:

```commandline
conda env create --file conda.yaml
conda activate dev
```

#### Скачать данные с DVC репозитория
```commandline
dvc pull
```

#### Обучение и тестирование модели с помощью DVC pipeline

```commandline
 dvc exp run
```

#### Метрики качества 
Текущие метрики качества на тестовой выборке находятся в файле:
```commandline
 dvcline/test/metrics.json
```

| mAP@50 | F1 score |
|--------|----------|
| 96.5 % | 98.3 %   |

