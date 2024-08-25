# Data Preparation
Here, we will proceed with a multi-turn GPT-2 chatbot model training tutorial using the [DailyDialog](http://yanran.li/dailydialog) dataset by default.
Please refer to the following instructions to utilize custom datasets.

### 1. DailyDialog Data
If you want to train on the DailyDialog dataset, simply set the `dailydialog_dataset_train` value in the `config/config.yaml` file to `True` as follows.
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
If you want to train your custom dataset, set the `dailydialog_dataset_train` value in the `config/config.yaml` file to `False` as follows.
You may require to implement your custom dataloader codes in `src/trainer/build.py`.
```yaml
dailydialog_train: False         # If True, dailydialog data will be loaded automatically.
dailydialog_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```