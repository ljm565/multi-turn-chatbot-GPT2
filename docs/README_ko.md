# Multi-turn Chatbot GPT-2

## Introduction
본 코드는 [DailyDialog](http://yanran.li/dailydialog) 데이터셋과 pre-trained GPT-2 바탕으로 multi-turn 챗봇 모델을 학습합니다.
본 프로젝트에 사용하는 GPT-2는 [Hugging Face GPT-2의 "gpt2" pre-trained 모델](https://huggingface.co/docs/transformers/model_doc/gpt2)입니다.
GPT-2를 이용한 multi-turn 챗봇 모델 학습에 대한 설명은 [GPT-2와 DailyDialog를 이용한 Multi-turn 챗봇 구현](https://ljm565.github.io/contents/gpt2.html)를 참고하시기 바랍니다.
<br><br><br>

## Supported Models
### Pre-trained GPT-2
* Hugging Face의 pre-trained GPT-2.
<br><br><br>


## Supported Tokenizer
### Pre-trained GPT-2 Tokenizer
* Hugging Face의 pre-trained GPT-2 토크나이저.
<br><br><br>

## 사용 데이터
* 실험으로 사용하는 데이터는 [DailyDialog](http://yanran.li/dailydialog) 데이터셋입니다.
* Custom 데이터를 사용할 경우, train/validation/test 데이터 경로를 `config/config.yaml`에 설정해야하며, custom tokenizer, dataloader를 구성하여 `src/trainer/build.py`에 코드를 구현해야합니다.
<br><br><br>

## Supported Devices
* CPU, GPU, multi-GPU (DDP), MPS (for Mac and torch>=1.12.0)
<br><br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>


## Project Tree
본 레포지토리는 아래와 같은 구조로 구성됩니다.
```
├── configs                           <- Config 파일들을 저장하는 폴더
│   └── *.yaml
│
└── src      
    ├── models
    |   └── gpt2.py                   <- GPT-2 모델 wrapper
    |
    ├── run                   
    |   ├── chatting.py               <- 학습된 모델 live demo 실행 코드
    |   ├── train.py                  <- 학습 실행 파일
    |   └── validation.py             <- 학습된 모델 평가 실행 파일
    |
    ├── tools                   
    |   ├── tokenizers
    |   |   └── tokenizer.py          <- Tokenizer class
    |   ├── early_stopper.py          <- Early stopper class 파일
    |   ├── evaluator.py              <- Metric evaluator class 파일
    |   ├── model_manager.py          
    |   └── training_logger.py        <- Training logger class 파일
    |
    ├── trainer                 
    |   ├── build.py                  <- Dataset, dataloader 등을 정의하는 파일
    |   └── trainer.py                <- 학습, 평가 등을 수행하는 class 파일
    |
    └── uitls                   
        ├── __init__.py               <- Logger, 버전 등을 초기화 하는 파일
        ├── data_utils.py             <- Dataloader 정의 파일
        ├── filesys_utils.py       
        ├── func_utils.py       
        └── training_utils.py     
```
<br><br>

## Tutorials & Documentations
Multi-turn 챗봇 모델을 학습하기 위해서 다음 과정을 따라주시기 바랍니다.
1. [Getting Started](./1_getting_started_ko.md)
2. [Data Preparation](./2_data_preparation_ko.md)
3. [Training](./3_trainig_ko.md)
4. ETC
   * [Evaluation](./4_model_evaluation_ko.md)
   * [Live Demo](./5_live_demo_ko.md)

<br><br><br>


## Training Results
### Results of GPT-2-based Multi-turn Chatbot Model
* BLEU Score History<br>
<img src="figs/bleu.jpg" width="80%"><br><br>

* NIST Score History<br>
<img src="figs/nist.jpg" width="80%"><br><br>

* Test Set Scores<br>
Validation set metric이 가장 높았던 모델을 test set에 적용한 결과.
    * BLEU-2: 0.4052
    * BLEU-4: 0.2268
    * NIST-2: 5.1797
    * NIST-4: 5.5162
    <br><br>

### Multi-turn GPT-2 Dialogue Results
"Q"는 사용자 메시지를 의미하고, "A"는 chatbot의 응답을 의미합니다.
```
Q: Have you ever been to France
A: Oh, yes! I went there once last summer.
Q: How was there?
A: It was amazing.
Q: Which spot was your best?
A: That of the Louvre.

Q: Did you see her yesterday?
A: Yes, I did. I saw her with a pretty girl.
Q: Oh where did you see her?                       
A: She was in a cafe. She told me she was going to a concert.
Q: I see. I've tried to contact her by phone, but she never received my call.
A: I'm sure she was in New York. She said she'd be in New York at the latest.
Q: Really? Thank you for your information. I will contact her today. Thanks.
```

<br><br><br>