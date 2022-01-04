import unittest

from utils.persistence import load_hparams_from_checkpoint_name

class MyTestCase(unittest.TestCase):
    def test_hparam_loading(self):
        chkpoint_name = "mnist_k1_L1_bs200_VAE_5.pkl"
        hparams = load_hparams_from_checkpoint_name(chkpoint_name)
        assert hparams == {
                "dataset_name": "mnist",
                "k": 1,
                "num_layers": 1,
                "batch_size": 200,
                "model_type": "VAE",
                "epoch": 5
            }


if __name__ == '__main__':
    unittest.main()
