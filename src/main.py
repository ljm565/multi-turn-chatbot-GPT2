import torch
import pickle
from argparse import ArgumentParser
import os
from train import Trainer
from utils.config import Config
import json
import sys

from utils.utils_func import check_data, make_dataset_path, preprocessing_query



def main(config_path:Config, args:ArgumentParser):
    device = torch.device('cuda:0') if args.device == 'gpu' else torch.device('cpu')
    print('Using {}'.format(device))

    if (args.cont and args.mode == 'train') or args.mode != 'train':
        try:
            config = Config(config_path)
            config = Config(config.base_path + '/model/' + args.name + '/' + args.name + '.json')
            base_path = config.base_path
        except:
            print('*'*36)
            print('There is no [-n, --name] argument')
            print('*'*36)
            sys.exit()
    else:
        config = Config(config_path)
        base_path = config.base_path

        # make neccessary folders
        os.makedirs(base_path+'model', exist_ok=True)
        os.makedirs(base_path+'loss', exist_ok=True)
        os.makedirs(base_path+'data/dailydialog/processed', exist_ok=True)
    
        # save processed data
        check_data(base_path)

        # split dataset path
        config.dataset_path = make_dataset_path(base_path)
        
        # redefine config
        config.loss_data_path = base_path + 'loss/' + config.loss_data_name + '.pkl'

        # make model related files and folder
        model_folder = base_path + 'model/' + config.model_name
        config.model_path = model_folder + '/' + config.model_name + '.pt'
        model_json_path = model_folder + '/' + config.model_name + '.json'
        os.makedirs(model_folder, exist_ok=True)
          
        with open(model_json_path, 'w') as f:
            json.dump(config.__dict__, f)
    
    trainer = Trainer(config, device, args.mode, args.cont)

    if args.mode == 'train':
        loss_data_path = config.loss_data_path
        print('Start training...\n')
        loss_data = trainer.training()

        print('Saving the loss related data...')
        with open(loss_data_path, 'wb') as f:
            pickle.dump(loss_data, f)

    elif args.mode == 'inference':
        print('Start inferencing...\n')
        trainer.inference('test', config.result_num)
            
    elif args.mode == 'chatting':
        msg = """
        Chatbot starts..
        If you want to chat new topic, enter the "new()"..
        If you want to exit, enter the "exit()"..
        
        Please enter the 3 multi-turn dialogues..
        """
        print(msg + '\n')

        tokenizer = trainer.tokenizer
        queries, turn, state = [], 0, 'A'
        while 1:
            turn += 1

            # for initializing topic of dialogues
            if len(queries) < 3:
                while len(queries) < 3:
                    state = 'A' if state == 'Q' else 'Q'
                    query = input(state + str(turn) + ': ')
                    queries.append(query)
                    turn += 1
                    if query == 'new()':
                        queries, turn, state = [], 0, 'A'
                        print('Please enter new 3 multi-turn dialogues..\n')
                        continue
                    elif query == 'exit()':
                        break
            
            # for multi-turn chatting
            else:
                state = 'A' if state == 'Q' else 'Q'
                query = input(state + str(turn) + ': ')
                queries.append(query)
                turn += 1
                if query == 'new()':
                    queries, turn, state = [], 0, 'A'
                    print('Please enter new 3 multi-turn dialogues..\n')
                    continue
                elif query == 'exit()':
                    break

            # query preprocessing
            query = preprocessing_query(queries, tokenizer)
            
            # answer of the model
            state = 'A' if state == 'Q' else 'Q'
            query = trainer.chatting(query)
            queries.append(query)
            print(state + str(turn) + ': ' + query )

        print('Chatbot ends..\n')

    else:
        print("Please select mode among 'train', 'inference', and 'chatting'..")
        sys.exit()



if __name__ == '__main__':
    path = os.path.realpath(__file__)
    path = path[:path.rfind('/')+1] + 'config.json'    

    parser = ArgumentParser()
    parser.add_argument('-d', '--device', type=str, required=True, choices=['cpu', 'gpu'])
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'inference', 'chatting'])
    parser.add_argument('-c', '--cont', type=int, default=0, required=False)
    parser.add_argument('-n', '--name', type=str, required=False)
    args = parser.parse_args()

    main(path, args)