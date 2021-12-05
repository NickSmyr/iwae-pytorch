import unittest

from torch.nn import Tanh, Linear

from modules.iwae import IWAE


class TestIWAEModule(unittest.TestCase):

    def setUp(self) -> None:
        self.c_in, self.w_in, = 1, 28
        self.q_dim, self.p_dim = 64, 64
        self.hidden_dims = [1024, 512, 64]
        self.use_bias = True
        self.iwae = IWAE(c_in=self.c_in, w_in=self.w_in, q_dim=self.q_dim, p_dim=self.p_dim, use_bias=self.use_bias,
                         hidden_dims=self.hidden_dims)

    def test_encoder_architecture(self) -> None:
        encoder = self.iwae.encoder
        self.assertEqual(6, len(encoder))
        #   - 1st layer > Linear
        self.assertEqual(Linear, type(encoder[0]))
        self.assertEqual(encoder[0].bias is not None, self.use_bias)
        self.assertEqual(self.c_in * self.w_in ** 2, encoder[0].in_features)
        self.assertEqual(self.hidden_dims[0], encoder[0].out_features)
        #   - 1st layer > Activation
        self.assertEqual(Tanh, type(encoder[1]))
        #   - 2nd layer > Linear
        self.assertEqual(Linear, type(encoder[2]))
        self.assertEqual(encoder[2].bias is not None, self.use_bias)
        self.assertEqual(self.hidden_dims[0], encoder[2].in_features)
        self.assertEqual(self.hidden_dims[1], encoder[2].out_features)
        #   - 2nd layer > Activation
        self.assertEqual(Tanh, type(encoder[3]))
        #   - 3rd layer > Linear
        self.assertEqual(Linear, type(encoder[4]))
        self.assertEqual(encoder[4].bias is not None, self.use_bias)
        self.assertEqual(self.hidden_dims[1], encoder[4].in_features)
        self.assertEqual(self.hidden_dims[2], encoder[4].out_features)
        #   - 3rd layer > Activation
        self.assertEqual(Tanh, type(encoder[5]))

    def test_decoder_architecture(self) -> None:
        decoder = self.iwae.decoder
        self.assertEqual(6, len(decoder))
        #   - 1st layer > Linear
        self.assertEqual(Linear, type(decoder[0]))
        self.assertEqual(decoder[0].bias is not None, self.use_bias)
        self.assertEqual(self.p_dim, decoder[0].in_features)
        self.assertEqual(self.hidden_dims[2], decoder[0].out_features)
        #   - 1st layer > Activation
        self.assertEqual(Tanh, type(decoder[1]))
        #   - 2nd layer > Linear
        self.assertEqual(Linear, type(decoder[2]))
        self.assertEqual(decoder[2].bias is not None, self.use_bias)
        self.assertEqual(self.hidden_dims[2], decoder[2].in_features)
        self.assertEqual(self.hidden_dims[1], decoder[2].out_features)
        #   - 2nd layer > Activation
        self.assertEqual(Tanh, type(decoder[3]))
        #   - 3rd layer > Linear
        self.assertEqual(Linear, type(decoder[4]))
        self.assertEqual(decoder[4].bias is not None, self.use_bias)
        self.assertEqual(self.hidden_dims[1], decoder[4].in_features)
        self.assertEqual(self.hidden_dims[0], decoder[4].out_features)
        #   - 3rd layer > Activation
        self.assertEqual(Tanh, type(decoder[5]))

    def tearDown(self) -> None:
        if self.iwae is not None:
            del self.iwae
