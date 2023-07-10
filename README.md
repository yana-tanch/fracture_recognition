###   Обучение модели детекции и классификации переломов на медицинских изображениях

### 

#### Установка conda окружения 
Для создания и активации conda окружения необходимо выполнить:

```commandline
conda env create --file conda.yaml
conda activate dev
```

#### Предобработка данных
```commandline
python src/prepare_dataset.py
```
