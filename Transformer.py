import tensorflow as tf
from tensorflow.python.keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        angle = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle

    def positional_encoding(self, position, d_model):
        # shape of [position, 1], shape of [1, d_model] --> [position, d_model // 2]
        angle_rads = self.get_angles(tf.range(position)[:, tf.newaxis], tf.range(d_model)[tf.newaxis, :], d_model)

        # apply sin to even indices in the array; 2i
        sine = tf.math.sin(
            angle_rads[:, 0::2])  # select all rows and every other column start from column 0 (even column)

        # apply cos to odd indices in the array; 2i+1
        cosine = tf.math.cos(
            angle_rads[:, 1::2])  # select all rows and every other column start from column 1 (odd column)

        # Here we simply concatenate sine values and cosine values instead of interweave them. Actually, it does not
        # really matter to our model since transformer does not inherently understand the concept of position,
        # as long as, each position gets a different PE encoding.
        pos_encoding = tf.concat([sine, cosine], axis=-1)  # generate a [position, d_model] tensor
        pos_encoding = pos_encoding[tf.newaxis, ...]  # [1, position * 2, d_model]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def feed_forward_network(d_model, dff=2048):
    # we first expand the matrix to size of dff and shrink back to d_model with an activation function ReLU in between
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # add a large negative number to mask the data

    # softmax logits
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (batch_size, seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0  # we want d_k to be an integer (d_k = d_model / num_heads)

        self.d_k = d_model / num_heads

        # Dense layer, linearly transform embedded tokens into Q,K,V matrices
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        # Linear layer used at the end of the multi-head attention to  transform concatenated data to d_model size
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # split the last dimension into (num_heads, k_d).
        x = tf.reshape(x, (batch_size, -1, self.num_heads, int(self.d_k)))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        # linear transformation
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_q, depth)

        # calculate the scaled dot product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        # Pass through linear layer and transform back to d_model size
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


class AddNorm(layers.Layer):
    def __init__(self, norm_epsilon=1e-6, dropout_rate=0.1):
        super(AddNorm, self).__init__()
        self.dropout = layers.Dropout(dropout_rate)  # probability that data will be dropout(set to zero)
        # adding a small epsilon to avoid dividing by zero
        self.norm = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

    def call(self, inputs, training=False):
        return self.norm(inputs + self.dropout(inputs, training=training))


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.norm_layer1 = AddNorm(dropout_rate=rate)
        self.norm_layer2 = AddNorm(dropout_rate=rate)

    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        norm_output = self.norm_layer1(x + attn_output, training=training)
        ffn_output = self.ffn(norm_output)
        output = self.norm_layer2(norm_output + ffn_output, training=training)
        return output


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.norm_layer1 = AddNorm(dropout_rate=rate)
        self.norm_layer2 = AddNorm(dropout_rate=rate)
        self.norm_layer3 = AddNorm(dropout_rate=rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask=None):
        attn_output, attn_weight1 = self.mha1(x, x, x, look_ahead_mask)
        norm_output = self.norm_layer1(x + attn_output, training=training)

        attn_output2, attn_weight2 = self.mha2(norm_output, encoder_output, encoder_output, padding_mask)
        norm_output2 = self.norm_layer2(norm_output + attn_output2, training=training)

        ffn_output = self.ffn(norm_output2)
        output = self.norm_layer3(norm_output2 + ffn_output, training=training)
        return output, attn_weight1, attn_weight2


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate=rate)

    def call(self, x, training, mask=None):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # This scaling is done because for large values of depth, the dot product grows large in magnitude, pushing the
        # softmax function where it has small gradients resulting in a very hard softmax. To counteract this,
        # we scale the dot products by the square root of their dimensionality.
        x = self.pos_encoding(x)
        # select everything in 1st dimension, up to seq_len in 2nd dimension, everything in 3rd dimension

        x = self.dropout(x, training=training)  # empirical technique to reduce overfitting
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate=rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights  # x = (batch_size, target_seq_len, d_model)


class FinalLayer(layers.Layer):
    def __init__(self, target_vocab_size=32000):
        super(FinalLayer, self).__init__()
        self.linear = tf.keras.layers.Dense(target_vocab_size)  # hyperparameter based on size of corpus
        self.softmax = layers.Softmax(axis=-1)

    def call(self, x):
        linear_output = self.linear(x)
        return self.softmax(linear_output)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)  # Ensure step is float
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
