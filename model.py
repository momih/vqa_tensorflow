from ops import conv2d_layer, fc_layer
import tensorflow as tf
import numpy as np
from resnet import ResNetFeatures

class Model(object):
    """
        TF implementation of "Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]
        [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, state_size, lr, batch_size, epochs, features_path, 
                 vocabulary_size, embedding_size=300, most_freq_limit=3000,
                 summary_dir='./logs/'):

        """

        :type max_ques_length: object
        :type embedding_size: object
        """
        self.state_size = state_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
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

        image_features = self.extract_resnet_features(self.images)
        text_features = self.extract_text_features(self.questions)
        attention_features = self.compute_attention(image_features, text_features)

        fc1 = tf.nn.relu(fc_layer(attention_features, 1024, name="fc1"))
        answer_logits = fc_layer(fc1, 3000, name="fc2")
        self.answer_prob = tf.nn.softmax(answer_logits)
        self.loss = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.answers, 
                                                                           logits=answer_logits))

    def _initialize_vars_and_summaries(self):
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)
        self.tf_summary_writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)

    def train(self):
        self._initialize_vars_and_summaries()
        optim = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999)\
                        .minimize(self.loss)
        
        pass

    def extract_resnet_features(self, img):
        resnet = ResNetFeatures()
        return resnet(img)
    
    def extract_text_features(self, text):
        word_embeddings = tf.get_variable('word_embeddings', [self.vocabulary_size, self.embedding_size])
        lstm_input = tf.nn.embedding_lookup(word_embeddings, text)
        _, final_state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(1024),
                                           lstm_input, dtype=tf.float32)
        return final_state.c


    def compute_attention(self, image, text):        
        text_replicated = self._replicate_state(text, (1, 14, 14, 1))
        
        # Now both the features from the resnet and lstm are concatenated along the depth axis
        features = tf.concat([image, text_replicated], axis=3)
        conv1 = tf.nn.relu(conv2d_layer(features, filters=512, kernel_size=(1,1), name="conv1"))
        conv2 = conv2d_layer(conv1, filters=2, kernel_size=(1,1), name="conv2")
        
        # Flatenning each attention map to perform softmax
        attention_map = tf.reshape(conv2, (self.batch_size, 14*14, 2))
        attention_map = tf.nn.softmax(attention_map, axis=1, name = "attention_map")
        image = tf.reshape(image, (self.batch_size, 192, 2048, 1))
        attention = tf.tile(tf.expand_dims(attention_map, 2), (1, 1, 2048, 1))
        image = tf.tile(image,(1,1,1,2))
        weighted = image * attention
        weighted_average = tf.reduce_mean(weighted, 1)
        
        # Flatten both glimpses into a single vector
        weighted_average = tf.reshape(weighted_average, (self.batch_size, 2048*2))
        attention = tf.concat([weighted_average, text], 1)
        return attention


    def _replicate_features(self, input_features, multiples, project=False):
        if not project:
            # Expanding dimensions of LSTM features to 4-D
            batch, features = input_features.shape
            x = tf.reshape(input_features, (batch, 1, 1, features))
            replicated = tf.tile(x, multiples)
        else:
            # TODO write deconv network to project LSTM features to resnet output
            pass
        return replicated
