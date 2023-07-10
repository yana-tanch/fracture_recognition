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

#### Обучение модели с помощью DVC

```commandline
 dvc exp run -f -n train
```