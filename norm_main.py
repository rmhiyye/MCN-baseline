import os
import time
import pickle

import src.config as config
from src.norm import n2c2Trainer
from src.utils import eval_map

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def main(config):
        
        start_time = time.time()

        # Load data
        n2c2_path = 'embedding_matrix/n2c2test_encode.pickle'
        with open(n2c2_path, 'rb') as f:
                test_dataset = pickle.load(f)

        trainer = n2c2Trainer(config, test_dataset)
        predictions, mentions = trainer.inference()

        inference_time_minutes = (time.time() - start_time) / 60

        print("Finish inference...")
        print("Inference time: {:.2f} minutes".format(inference_time_minutes))

        MAP_k1, class_MAP_k1 = eval_map(test_dataset, predictions, mentions)
        print('MAP', MAP_k1)
        for idx in class_MAP_k1.keys():
                print(f'Word length: {idx}, MAP: {class_MAP_k1[idx]["MAP"]}, Size: {class_MAP_k1[idx]["size"]}')
        MAP_k5, class_MAP_k5 = eval_map(test_dataset, predictions, mentions, k=5)
        print('MAP@5', MAP_k5)
        for idx in class_MAP_k5.keys():
                print(f'Word length: {idx}, MAP: {class_MAP_k5[idx]["MAP"]}, Size: {class_MAP_k5[idx]["size"]}')
        
        print("Done.")
        
if __name__ == '__main__':
        main(config)