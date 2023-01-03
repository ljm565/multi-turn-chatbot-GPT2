# Multi-turn Chatbot GPT-2
<!-- ## 설명
본 코드는 Google Play Store Apps 리뷰 데이터 바탕으로 pre-trained BERT 모델을 fine-tuning하여 긍정, 보통, 부정 감성 분류 모델을 제작합니다.
본 프로젝트에 사용한 BERT 모델은 [Hugging Face BERT의 "bert-base-uncased" pre-trained 모델](https://huggingface.co/docs/transformers/model_doc/bert)입니다.
BERT 기반 감성 분류 모델과 pre-trained BERT 사용에 대한 설명은 [Pre-trained BERT Fine-tuning을 통한 Google Play Store Apps 리뷰 감성 분류](https://ljm565.github.io/contents/bert3.html)를 참고하시기 바랍니다.
<br><br><br>

## 모델 종류
* ### Pre-trained BERT
    긍정, 보통, 부정의 3가지 감정을 분류하기 위해 Hugging Face pre-trained BERT를 fine-tuning 합니다.
<br><br><br>


## 토크나이저 종류
* ### Wordpiece Tokenizer
    Hugging Face의 pre-trained wordpiece 토크나이저를 사용합니다.
<br><br><br>

## 사용 데이터
* 실험으로 사용하는 데이터는 [Google Play Store App review](https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/) 데이터입니다. 5점 만점 평점 중, 3점 미만은 부정, 3점 초과는 긍정, 3점은 보통으로 분류하여 사용합니다.
<br><br><br>


## 사용 방법
* ### 학습 방법
    학습을 시작하기 위한 argument는 4가지가 있습니다.<br>
    * [-d --device] {cpu, gpu}, **필수**: 학습을 cpu, gpu로 할건지 정하는 인자입니다.
    * [-m --mode] {train, test}, **필수**: 학습을 시작하려면 train, 학습된 모델을 가지고 있어서 loss, accuracy, sample을 보고싶은 경우에는 test로 설정해야합니다. test를 사용할 경우, [-n, --name] 인자가 **필수**입니다.
    * [-c --cont] {1}, **선택**: 학습이 중간에 종료가 된 경우 다시 저장된 모델의 체크포인트 부분부터 학습을 시작할 수 있습니다. 이 인자를 사용할 경우 -m train 이어야 합니다. 
    * [-n --name] {name}, **선택**: 이 인자는 -c 1 혹은 -m test 경우 사용합니다.
    중간에 다시 불러서 학습을 할 경우 모델의 이름을 입력하고, test를 할 경우에도 test 할 모델의 이름을 입력해주어야 합니다(최초 학습시 src/config.json에서 정한 모델의 이름의 폴더가 형성되고 그 폴더 내부에 모델 및 모델 파라미터가 json 파일로 형성 됩니다).<br><br>

    터미널 명령어 예시<br>
    * 최초 학습 시
        ```
        python3 src/main.py -d cpu -m train
        ```
    * 중간에 중단 된 모델 이어서 학습 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/src/config.json이 아닌, base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.
        ```
        python3 src/main.py -d gpu -m train -c 1 -n {model_name}
        ```
    * 최종 학습 된 모델의 test set에 대한 loss, accuracy를 확인할 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/src/config.json이 아닌, base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 수정사항이 반영됩니다.
        ```
        python3 src/main.py -d cpu -m test -n {model_name}
        ```
    <br><br>

* ### 모델 학습 조건 설정 (config.json)
    **주의사항: 최초 학습 시 config.json이 사용되며, 이미 한 번 학습을 한 모델에 대하여 parameter를 바꾸고싶다면 base_path/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.**
    * pretrained_model: bert-base-uncased, bert-base-cased, bert-large-uncased 등 pre-trained BERT 모델 선택.
    * base_path: 학습 관련 파일이 저장될 위치.
    * model_name: 학습 모델이 저장될 파일 이름 설정. 모델은 base_path/src/model/{model_name}/{model_name}.pt 로 저장.
    * loss_data_name: 학습 시 발생한 loss data를 저장하기 위한 이름 설정. base_path/src/loss/{loss_data_name}.pkl 파일로 저장. 내부에 중단된 학습을 다시 시작할 때, 학습 과정에 발생한 loss 데이터를 그릴 때 등 필요한 데이터를 dictionary 형태로 저장.
    * max_len: 토큰화 된 리뷰 데이터의 최대 길이.
    * n_class: 분류할 카테고리 수. 현재는 긍정, 보통, 부정이므로 3.
    * batch_size: batch size 지정.
    * epochs: 학습 epoch 설정.
    * lr: learning rate 지정.
    * early_stop_criterion: validation set의 최대 accuracy를 내어준 학습 epoch 대비 설정된 숫자만큼 epoch이 지나도 나아지지 않을 경우 학습 조기 종료.
    * result_num: 모델 테스트 시, 결과를 보여주는 sample 개수.
    <br><br><br>


## 결과
* ### Sentiment Classification BERT 모델 결과
    * Loss History<br>
    <img src="images/loss.png" width="80%"><br><br>

    * Accuracy History<br>
    <img src="images/acc.png" width="80%"><br><br>
    
    * 최종 Test Set Statistics<br>
        Validation set accuracy가 가장 높았던 11 epoch 모델을 test set에 적용한 결과<br><br>
        Acc: 0.86<br>
        ```
                      precision    recall  f1-score   support

            negative       0.84      0.92      0.88       514
            mediocre       0.84      0.77      0.81       529
            positive       0.91      0.90      0.90       533

            accuracy                           0.86      1576
           macro avg       0.86      0.86      0.86      1576
        weighted avg       0.86      0.86      0.86      1576
        ```
        <img src="images/statistics.png" width="100%"><br><br>

    * 결과 샘플<br>
        ```
        # sample 1
        review: [CLS] i had paid once for this app and had login to it. now i have another mobile and want to use my acount on this device, but this app asket to pay first before login. should i pay each time i change my device? [SEP]
        gt    : negative
        pred  : negative


        # sample 2
        review: [CLS] i got this app to track my medication and it's perfect! i can set up how i want to take each medicine ( yes / no or quantity ), see the start date and adherence in the summary view, and even track side effects and effectiveness each day then see them in a chronological list in the sunmary. and the best part is that it's not tied to some medical database! added bonus : i can track real to - dos as well. overall, love this app! [SEP]
        gt    : positive
        pred  : positive


        # sample 3
        review: [CLS] great app [SEP]
        gt    : positive
        pred  : positive
        ```
    <br><br>



<br><br><br> -->