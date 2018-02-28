from ops import *
import tensorflow as tf
import numpy as np
import resnet

class VQA(object):
    """
        TF implementation of "Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]
        [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, state_size, lr, batch_size, epochs, features_path, most_freq_limit=3000,
                 max_ques_length=15, summary_dir='./logs/'):

        """

        :type max_ques_length: object
        :type embedding_size: object
        """
        self.max_ques_length = max_ques_length
        self.state_size = state_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.most_freq_limit = most_freq_limit
        self.summary_dir = summary_dir
        self.features_path = features_path
        self.session = tf.Session()
        self.build_model()


    def build_model(self):
        # Creating placeholders for the question and the answer
        self.questions = tf.placeholder(tf.float32, shape=[None, self.max_ques_length, 1])
        self.answers = tf.placeholder(tf.float16, shape=[None, self.most_freq_limit])
        self.images = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])

        image_features = resnet.extract_features(self.images)

        # Extracting features from the question
        with tf.variable_scope("txt_features") as scope:
            _, final_state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(1024), self.question,
                                         dtype=tf.float32)
            text_features = self._replicate_state(final_state.c, (14, 14, 2048))

        # Now both the features from the resnet and lstm are concatenated along the depth axis
        features = tf.concat([image_features, tiled], axis=2)
        conv1 = conv_layer(features, filters=512, kernel_size=(1,1), name="conv1", activation="relu")
        conv2 = conv_layer(first_conv, filters=2, kernel_size=(1,1), name="conv2", activation="softmax")

        weighted_average = self.compute_attention([image_features, second_conv])

        attention_features = tf.concat([tiled, weighted_average])
        fc1 = fc_layer(attention_features, 1024, activation="relu", name="fc1")
        answer_pred = fc_layer(fc1, 3000, activation="sigmoid", name="fc2")

        # TODO need to correct loss function
        self.loss = tf.reduce_mean(-tf.log(answer_pred))

    def _initialize_vars_and_summaries(self):
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)
        self.tf_summary_writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)

    def train(self):
        optim = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999)\
                        .minimize(self.loss)
        pass


    def compute_attention(self):
        # TODO
        pass


    def _replicate_features(self, input_features, multiples, project=False):
        if not project:
            replicated = tf.tile(input_features, multiples)
        else:
            # TODO write deconv network to project LSTM features to resnet output
            pass