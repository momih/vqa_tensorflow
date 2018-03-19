from resnet_utils import resnet_arg_scope
from resnet import resnet_v2_152
from VQA import VQADataSet
import tensorflow as tf
import time
import os

# TODO add summaries
# TODO add validation

class Model(object):
    """
        TF implementation of "Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]
        [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, batch_size, init_lr=0.001, reuse=False, vocabulary_size=None, state_size=1024,
                 embedding_size=300, dropout_prob=0.5, most_freq_limit=3000,
                 summary_dir='./logs/', resnet_weights_path = 'resnet_ckpt/resnet_v2_152.ckpt', 
                 project=False):

        """

        :type max_ques_length: object
        :type embedding_size: object
        """
        self.state_size = state_size
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.reuse= reuse
        self.embedding_size = embedding_size
        self.data = VQADataSet()
        self.vocabulary_size = self.data.vocab_size if vocabulary_size is None else vocabulary_size
        self.dropout_prob = dropout_prob
        self.most_freq_limit = most_freq_limit
        self.summary_dir = summary_dir
        self.resnet_weights_path = resnet_weights_path
        self.project = project
        self.sess = tf.Session()
        self.build_model()
        self._check_resnet_weights()

    def build_model(self):
        print('\nBuilding Model')
        # Creating placeholders for the question and the answer
        self.questions = tf.placeholder(tf.int64, shape=[None, 15], name="question_vector") 
        self.answers = tf.placeholder(tf.float32, shape=[None, self.most_freq_limit], name="answer_vector")
        self.images = tf.placeholder(tf.float32, shape=[None, 448, 448, 3], name="images_matrix")
        

        arg_scope = resnet_arg_scope()
        with tf.contrib.slim.arg_scope(arg_scope):
            resnet_features, _ = resnet_v2_152(self.images, reuse=tf.AUTO_REUSE)
        depth_norm = tf.norm(resnet_features, ord='euclidean', keepdims=True, axis=3) + 1e-8
        self.image_features = resnet_features/depth_norm
        
        with tf.variable_scope("text_features") as scope:
            if self.reuse:
                scope.reuse_variables()
            self.word_embeddings = tf.get_variable('word_embeddings', 
                                              [self.vocabulary_size,
                                               self.embedding_size],
                                               initializer=tf.contrib.layers.xavier_initializer())
            word_vectors = tf.nn.embedding_lookup(self.word_embeddings, self.questions)
            len_word = self._len_seq(word_vectors)
            
            embedded_sentence = tf.nn.dropout(tf.nn.tanh(word_vectors, name="embedded_sentence"),
                                       keep_prob=self.dropout_prob)
            lstm = tf.nn.rnn_cell.LSTMCell(self.state_size,
                                           initializer=tf.contrib.layers.xavier_initializer())
            _, final_state = tf.nn.dynamic_rnn(lstm, embedded_sentence,
                                               sequence_length=len_word,
                                               dtype=tf.float32)
            self.text_features = final_state.c
        
        self.attention_features = self.compute_attention(self.image_features,
                                                         self.text_features)
        
        with tf.variable_scope("fully_connected") as scope:
            if self.reuse:
                scope.reuse_variables()
            self.fc1 = tf.nn.dropout(tf.nn.relu(self.fc_layer(self.attention_features, 1024, name="fc1")),
                                     keep_prob=self.dropout_prob)
            self.fc2 = self.fc_layer(self.fc1, 3000, name="fc2")
        
        self.answer_prob = tf.nn.softmax(self.fc2)            
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.answers, 
                                                                              logits=self.fc2))
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        self.inc = tf.assign_add(self.global_step, 1, name='increment')
        self.lr = tf.train.exponential_decay(learning_rate=self.init_lr, 
                                             global_step=self.global_step,
                                             decay_steps=10000,
                                             decay_rate=0.5,
                                             staircase=True)
        
        self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999, name="optim")


    def train(self, epochs):
        self.saver = tf.train.Saver()
        self.tf_summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
       
        # Loading resnet pretrained weights        
        resnet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet")
        load_resnet = tf.train.Saver(var_list=resnet_vars)
        load_resnet.restore(self.sess, self.resnet_weights_path)
        
        # Freezing resnet weights
        train_vars = [x for x in tf.global_variables() if "resnet" not in x.name]
        train_step = self.optimizer.minimize(self.loss, var_list=train_vars, 
                                             global_step=self.global_step)
        
        # Initializing all variables
        print('Initializing variables')
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        self.data.encode_into_vector()
        
        start_time = time.time()
        print('Starting training')        
        for epoch in range(epochs):
            steps = 10#data.number_of_questions // self.batch_size
            for idx in range(steps):
                print("Step {:4d} of epoch {:2d}".format(idx, epoch))
                print('\nGetting batches')
                q, a, img = self.data.next_batch(self.batch_size)
                vqa_dict = {self.questions: q, self.answers: a, self.images: img}
                _, cost, _a = self.sess.run([train_step, self.loss, self.inc], feed_dict=vqa_dict)
                
                print("\nEpoch: [%2d] [%4d/%4d] time: %4.4f, Loss: %.8f"
                    % (epoch, idx, steps,
                       time.time() - start_time, cost))


    def compute_attention(self, image, text): 
        with tf.variable_scope("attention") as scope:
            if self.reuse:
                scope.reuse_variables()
            text_replicated = self._replicate_features(text, (1, 14, 14, 1), 
                                                       project=self.project)
            
            # Now both the features from the resnet and lstm are concatenated along the depth axis
            features = tf.nn.dropout(tf.concat([image, text_replicated], axis=3), 
                                     keep_prob=self.dropout_prob)
            conv1 = tf.nn.dropout(self.conv2d_layer(features, filters=512, 
                                               kernel_size=(1,1), 
                                               name="attention_conv1"),
                                  keep_prob=self.dropout_prob)
            conv2 = self.conv2d_layer(conv1, filters=2, kernel_size=(1,1), name="attention_conv2")
            
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
            attention_output = tf.nn.dropout(tf.concat([weighted_average, text], 1), self.dropout_prob)
        return attention_output
    
    def conv2d_layer(self, input_tensor, filters, kernel_size=(3,3), stride=1, name="conv", padding='VALID'):
        with tf.variable_scope(name):
            weights = tf.get_variable('conv_weights', [kernel_size[0], kernel_size[1], input_tensor.get_shape()[-1], filters],
                                initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('conv_bias', [filters], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(input_tensor, weights, strides=[1, stride, stride, 1], padding=padding)
            conv = tf.nn.bias_add(conv, biases)
            return conv

    def deconv2d_layer(self, input_tensor, filters, output_size, 
                       kernel_size=(5,5), stride=2, name="deconv2d"):
        with tf.variable_scope(name):
            h, w = output_size
            weights = tf.get_variable('deconv_weights', 
                                shape=[kernel_size[0], kernel_size[1],
                                       filters, input_tensor.get_shape()[-1]],
                                initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', [filters], initializer=tf.constant_initializer(0.0))
            
            output_dims = [self.batch_size, h, w, filters]
            deconv = tf.nn.conv2d_transpose(input_tensor, weights, strides=[1, stride, stride, 1],
                                            output_shape=output_dims)
            deconv = tf.nn.bias_add(deconv, biases)
            return deconv
    
    def fc_layer(self, input_tensor, neurons, name="fc"):
        with tf.variable_scope(name):
            weights = tf.get_variable('fc_weights', [input_tensor.get_shape()[-1], neurons],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('fc_biases', [neurons], initializer=tf.constant_initializer(0.0))
            
            output = tf.matmul(input_tensor, weights) + biases
            return output

    def _len_seq(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length
    
    def _replicate_features(self, input_features, multiples, project=False):
        x = tf.reshape(input_features, (self.batch_size, 1, 1, self.state_size))
        if not project:
            # Expanding dimensions of LSTM features to 4-D
            replicated = tf.tile(x, multiples)
        else:
            dc1 = self.deconv2d_layer(x, 1024, output_size=(2,2), name="dc1")
            x1 = tf.nn.dropout(dc1, self.dropout_prob)
            
            dc2 = self.deconv2d_layer(x1, 1536, output_size=(4,4), name="dc2")
            x2 = tf.nn.dropout(dc2, self.dropout_prob)
            
            dc3 = self.deconv2d_layer(x2, 2048, output_size=(8,8), name="dc3")
            x3 = tf.nn.dropout(dc3, self.dropout_prob)
            
            dc4 = self.deconv2d_layer(x3, 2048, output_size=(16,16), name="dc4")
            x4 = tf.nn.dropout(dc4, self.dropout_prob)
            
            replicated = tf.nn.dropout(self.conv2d_layer(x4, 2048, kernel_size=(3,3), 
                                                         name="conv_dc4"), 0.5)            
        return replicated
    
    def _check_resnet_weights(self):
        resnet_dir = './resnet_ckpt'
        if not os.path.exists(resnet_dir):
            os.mkdir(resnet_dir)
            url = "http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz"
            os.system("wget " + url)
            command = 'tar -xvzf {} -C ./resnet_ckpt/'.format(url.split("/")[-1])
            os.system(command)
