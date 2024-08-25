# Data Preparation
여기서는 multi-turn GPT-2 챗봇 모델 훈련 튜토리얼을 위해 다음 카페의 [DailyDialog](http://yanran.li/dailydialog) 데이터셋을 사용합니다.
Custom 데이터를 이용하기 위해서는 아래 설명을 참고하시기 바랍니다.

### 1. DailyDialog Data
DailyDialog 데이터를 학습하고싶다면 아래처럼 `config/config.yaml` 파일의 `dailydialog_train` 값을 `True`로 설정하면 됩니다.
```yaml
dailydialog_train: True         # If True, dailydialog data will be loaded automatically.
dailydialog_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
만약 custom 데이터를 학습하고 싶다면 아래처럼 `config/config.yaml`의 `dailydialog_train` 값을 `False` 설정하면 됩니다.
다만 `src/trainer/build.py`에 custom dataset 사용을 위한 코드를 추가로 작성해야합니다.
```yaml
dailydialog_train: False         # If True, dailydialog data will be loaded automatically.
dailydialog_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```