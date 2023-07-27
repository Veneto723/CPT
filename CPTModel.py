import Transformer
from DataProcessor import process_text
import tensorflow as tf


class CPTModel(tf.keras.Model):
    def __init__(self, num_layer_enc, num_layer_dec, d_model, num_head, dff, input_vocab_size, output_vocab_size,
                 pe_input, pe_output, rate=0.1):
        super(CPTModel, self).__init__()

        self.encoder = Transformer.Encoder(num_layer_enc, d_model, num_head, dff, input_vocab_size, pe_input, rate)
        self.decoder_u = Transformer.Decoder(num_layer_dec, d_model, num_head, dff, output_vocab_size, pe_output, rate)
        self.decoder_g = Transformer.Decoder(num_layer_dec, d_model, num_head, dff, output_vocab_size, pe_output, rate)
        self.final_layer_u = Transformer.FinalLayer()
        self.final_layer_g = Transformer.FinalLayer()

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask, task='understanding'):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        if task == 'understanding':
            dec_output, attention_weights = self.decoder_u(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
            final_output = self.final_layer_u(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        else:
            dec_output, attention_weights = self.decoder_g(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
            final_output = self.final_layer_g(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


START_TOKEN, END_TOKEN = 101, 102


def evaluate(model, inp, tokenizer, max_seq_len=256, task='understanding'):
    # TODO not finished Convert to padded sequence
    seq = process_text(inp, tokenizer, max_seq_len)
    # convert to tensor
    input_tensor = tf.convert_to_tensor(seq)
    encoder_input = input_tensor
    decoder_input = [START_TOKEN] * len(encoder_input)
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_seq_len):
        enc_padding_mask, combined_mask, dec_padding_mask = Transformer.create_masks(encoder_input, output)

        predictions, attention_weights = model(encoder_input, output, False, enc_padding_mask, combined_mask,
                                               dec_padding_mask, task)

        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN):
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights
