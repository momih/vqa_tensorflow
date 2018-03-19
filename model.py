from ops import conv2d_layer, fc_layer
from resnet import resnet_v2_152
from resnet_utils import resnet_arg_scope
from VQA import VQADataSet
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
import time


slim = tf.contrib.slim

class Model(object):
    """
        TF implementation of "Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]
        [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, batch_size, init_lr=0.001, reuse=False, vocabulary_size=10546, state_size=1024,
                 embedding_size=300, dropout_prob=0.5, most_freq_limit=3000,
                 summary_dir='./logs/'):

        """

        :type max_ques_length: object
        :type embedding_size: object
        """
        self.state_size = state_size
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.reuse= reuse
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.dropout_prob = dropout_prob
        self.most_freq_limit = most_freq_limit
        self.summary_dir = summary_dir
        self.sess = tf.Session()
        self.build_model()

    def build_model(self):
        print('\nBuilding Model')
        # Creating placeholders for the question and the answer
        self.questions = tf.placeholder(tf.int64, shape=[None, 15], name="question_vector") 
        self.answers = tf.placeholder(tf.float32, shape=[None, self.most_freq_limit], name="answer_vector")
        self.images = tf.placeholder(tf.float32, shape=[None, 448, 448, 3], name="images_matrix")

        arg_scope = resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            self.image_features, _ = resnet_v2_152(self.images, reuse=tf.AUTO_REUSE)
        
        with tf.variable_scope("text_features") as scope:
            if self.reuse:
                scope.reuse_variables()
            word_embeddings = tf.Variable(tf.random_uniform([10546, self.embedding_size], 0.1, 1.0), name="embeddings")
            word_vectors = tf.nn.embedding_lookup(word_embeddings, self.questions)
            lstm_input = tf.nn.dropout(tf.nn.tanh(word_vectors, name="lstm_input"),
                                       keep_prob=self.dropout_prob)
            _, final_state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(self.state_size),
                                               lstm_input, dtype=tf.float32)
            self.text_features = final_state.c
        
        self.attention_features = self.compute_attention(self.image_features, self.text_features)

        self.fc1 = tf.nn.dropout(tf.nn.relu(fc_layer(self.attention_features, 1024, name="fc1")),
                                 keep_prob=self.dropout_prob)
        self.answer_logits = tf.nn.dropout(fc_layer(self.fc1, 3000, name="fc2"),
                                           keep_prob=self.dropout_prob)
        self.answer_prob = tf.nn.softmax(self.answer_logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.answers, 
                                                                           logits=self.answer_logits))
        
        self.global_step = 0
        self.lr = tf.train.exponential_decay(learning_rate=self.init_lr, 
                                             global_step=self.global_step,
                                             decay_steps=10000,
                                             decay_rate=0.5,
                                             staircase=True)
        
        self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999, name="optim")


    def train(self, epochs):
        # Loading resnet pretrained weights
        train_vars = [x for x in tf.global_variables() if "resnet" not in x.name]
        
        self.saver = tf.train.Saver()
        self.tf_summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
       
        init_op = tf.global_variables_initializer()
        
        resnet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet")
        load_resnet = tf.train.Saver(var_list=resnet_vars)
        load_resnet.restore(self.sess, 'resnet_ckpt/resnet_v2_152.ckpt')
         
        train_vars = [x for x in tf.global_variables() if "resnet" not in x.name]
        train_step = self.optimizer.minimize(self.loss, var_list=train_vars, 
                                             global_step=self.global_step)
        self.sess.run(init_op)
        self.data = VQADataSet()
        self.data.encode_into_vector()
        
        start_time = time.time()

        for epoch in range(epochs):
            steps = 10#data.number_of_questions // self.batch_size
            for idx in range(steps):
                print("\nStarting step {:4d} of epoch {:2d}".format(idx, epoch))
                print('\nGetting batches')
                q, a, img = self.data.next_batch(self.batch_size)
                vqa_dict = {self.questions: q, self.answers: a, self.images: img}
                _, cost = self.sess.run([train_step, self.loss], feed_dict=vqa_dict)
                
                self.global_step += 1
                print("\nEpoch: [%2d] [%4d/%4d] time: %4.4f, Loss: %.8f"
                    % (epoch, idx, steps,
                       time.time() - start_time, cost))


    def compute_attention(self, image, text): 
        with tf.variable_scope("attention") as scope:
            if self.reuse:
                scope.reuse_variables()
            text_replicated = self._replicate_features(text, (1, 14, 14, 1))
            
            # Now both the features from the resnet and lstm are concatenated along the depth axis
            features = tf.nn.dropout(tf.concat([image, text_replicated], axis=3))
            conv1 = tf.nn.dropout(conv2d_layer(features, filters=512, kernel_size=(1,1), name="attention_conv1"))
            conv2 = conv2d_layer(conv1, filters=2, kernel_size=(1,1), name="attention_conv2")
            
            # Flatenning each attention map to perform softmax
            attention_map = tf.reshape(conv2, (self.batch_size, 14*14, 2))
            attention_map = tf.nn.softmax(attention_map, axis=1, name = "attention_map")
            image = tf.reshape(image, (self.batch_size, 196, 2048, 1))
            attention = tf.tile(tf.expand_dims(attention_map, 2), (1, 1, 2048, 1))
            image = tf.tile(image,(1,1,1,2))
            weighted = image * attention
            weighted_average = tf.reduce_mean(weighted, 1)
            
            # Flatten both glimpses into a single vector
            weighted_average = tf.reshape(weighted_average, (self.batch_size, 2048*2))
            attention = tf.nn.dropout(tf.concat([weighted_average, text], 1), self.dropout_prob)
        return attention


    def _replicate_features(self, input_features, multiples, project=False):
        if not project:
            # Expanding dimensions of LSTM features to 4-D
            x = tf.reshape(input_features, (self.batch_size, 1, 1, self.state_size))
            replicated = tf.tile(x, multiples)
        else:
            # TODO write deconv network to project LSTM features to resnet output
            pass
        return replicated