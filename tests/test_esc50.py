import os
import unittest

from scripts import train


class TestsESC50(unittest.TestCase):

    def test_1_shot_1_way(self):
        config = {
            "data.dataset": "esc50",
            "data.split": "esc",
            "data.train_way": 1,
            "data.train_support": 1,
            "data.train_query": 1,
            "data.test_way": 1,
            "data.test_support": 1,
            "data.test_query": 1,
            "data.episodes": 10,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "128,47,1",
            "model.z_dim": 64,
            "train.epochs": 1,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": 'test_esc50_1way1shot.h5'
        }
        train(config)
        os.remove('test_esc50_1way1shot.h5')

    def test_5_shot_2_way(self):
        config = {
            "data.dataset": "esc50",
            "data.split": "esc",
            "data.train_way": 2,
            "data.train_support": 5,
            "data.train_query": 5,
            "data.test_way": 2,
            "data.test_support": 5,
            "data.test_query": 5,
            "data.episodes": 10,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "128,47,1",
            "model.z_dim": 64,
            "train.epochs": 10,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": 'test_mi_net.h5'
        }
        train(config)
        os.remove('test_mi_net.h5')

    def test_5_shot_5_way(self):
        config = {
            "data.dataset": "esc50",
            "data.split": "esc",
            "data.train_way": 5,
            "data.train_support": 5,
            "data.train_query": 5,
            "data.test_way": 5,
            "data.test_support": 5,
            "data.test_query": 5,
            "data.episodes": 3,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "128,47,1",
            "model.z_dim": 64,
            "train.epochs": 1,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": 'test_esc50_5way5shot.h5'
        }
        train(config)
        os.remove('test_esc50_5way5shot.h5')

    def test_10_shot_1_way(self):
        config = {
            "data.dataset": "esc50",
            "data.split": "esc",
            "data.train_way": 1,
            "data.train_support": 10,
            "data.train_query": 10,
            "data.test_way": 1,
            "data.test_support": 10,
            "data.test_query": 10,
            "data.episodes": 3,
            "data.cuda": False,
            "data.gpu": 0,
            "model.x_dim": "128,47,1",
            "model.z_dim": 64,
            "train.epochs": 1,
            'train.optim_method': "Adam",
            "train.lr": 0.001,
            "train.patience": 5,
            "model.save_path": 'test_mi_net.h5'
        }
        train(config)
        os.remove('test_mi_net.h5')


if __name__ == "__main__":
    unittest.main()
