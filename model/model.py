import numpy as np
import re
import json
import tensorflow as tf

class attention_model():
    def __init__(self,
                 max_len = 80,
                 num_classes = 2,
                 n_words = None,
                 latent_factor = 128,
                 num_hidden = 64,
                 learning_rate = 0.001):
        self.max_len = max_len
        self.num_classes = num_classes
        self.n_words = n_words
        self.latent_factor = latent_factor
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate

    def build_model(self):
        tf.reset_default_graph()
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            #Input layer 
            self.inputs_ = tf.placeholder(tf.int32, [None, self.max_len], name="input")           #here input size is [m, 80]
            self.labels_ = tf.placeholder(tf.int32, [None, self.num_classes], name="labels")      #label is binary 0 or 1
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')                         #for dropout probability

            #RNN layer
            #randomize weights within interval (0, 1) in embedding matrix of size [n_words + 1, 128]
            #This is really important as it sets all zeros for first row. Thus, it will have no effects on embedding.
            embedding_matrix_zero = tf.Variable(tf.random_uniform((1, self.latent_factor), 0, 0))
            embedding_matrix = tf.Variable(tf.random_uniform((self.n_words, self.latent_factor), 0, 1))
            embedding_matrix = tf.concat([embedding_matrix, embedding_matrix_zero], axis = 0) 
            #encoding word_index vector [1,80] of each review into word_embedding vector of each review [80, 128]
            embedding_result = tf.nn.embedding_lookup(embedding_matrix, self.inputs_)                               

            GRU_fw = tf.contrib.rnn.GRUCell(self.num_hidden)
            GRU_fw_Drop = tf.contrib.rnn.DropoutWrapper(GRU_fw, output_keep_prob=self.keep_prob)
            GRU_bw = tf.contrib.rnn.GRUCell(self.num_hidden)
            GRU_bw_Drop = tf.contrib.rnn.DropoutWrapper(GRU_bw, output_keep_prob=self.keep_prob)
    
            GRU_outputs, GRU_outputs_state = tf.nn.bidirectional_dynamic_rnn(GRU_fw_Drop, GRU_bw_Drop, embedding_result, dtype=tf.float32)
            GRU_outputs = tf.concat(GRU_outputs, axis = 2) #due to the bidirection process, the output is [m, 80, 128]
    
            #Attention layer: one layer is enough in this case
            attention_inputs = tf.reshape(GRU_outputs, [-1, 2*self.num_hidden])      #reshape GRU output into (m*80, 128) 
            self.attention_outputs = tf.layers.dense(attention_inputs, \
                                        1, activation=tf.nn.tanh)                     #output would be (m*80, 1)
            self.attention_outputs = tf.reshape(self.attention_outputs, [-1, self.max_len, 1]) #reshape to (m, 80, 1)
            self.context_weights = tf.nn.softmax(self.attention_outputs, axis = 1)        #weight shape would be (m, 80, 1) #, 
            self.weighted_words_vec = tf.reduce_sum(GRU_outputs * self.context_weights, axis=1) 
    
            #prediction layer
            self.logits = tf.layers.dense(self.weighted_words_vec, 2, activation=None)
            self.prediction = tf.nn.softmax(self.logits)

            #Optimization rule
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels_))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            #Evaluation through accuracy
            self.correct_pred = tf.equal(tf.cast(tf.round(self.prediction), tf.int32), self.labels_)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.saver = tf.train.Saver()
    
    def get_batches(self, x, y, batch_size):
        """
        Arguments:
        x -- feature
        y -- label
        batch_size -- number of samples in one batch 
    
        Returns:
        generator -- generate batches for training loop
        """   
        n_batches = len(x)//batch_size
        x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
        for index in range(0, len(x), batch_size):
            yield x[index:index+batch_size], y[index:index+batch_size]
    
    def top_n_feature(self, inputs, context_weights, n_features):
        """
        Arguments:
        inputs -- input word_index_vector with size of [m, latent factor] 
        context_weights -- weights(alpha) for contents from attention layer
        n_features -- number of top n important features 
    
        Returns:
        key_features -- top n important feature. e.g. n = 2 -> ['很','好']
        """   
        m = len(context_weights)
        key_features = []
        for i in range(m):
            words = list(inputs[i])
            weights = list(context_weights[i].flatten())
            key_features.append([words[i] for i in np.argsort(weights)[::-1][:n_features]])
        return key_features
    
    def index_to_word(self, inputs, word_dict):
        """
        Arguments:
        inputs -- input word_index_vector with size of [m, 80] 
        word_dict -- word-index dictionary
    
        Returns:
        words -- back convert index to corresponding words. e.g [2, 0, 3, 41....0] -> ['东西','很','好']
        """   
        m = len(inputs)
        inverse = {}
        for key, value in word_dict.items():
            inverse[value] = key
        words = []
        for i in range(m):
            word = [inverse[index] for index in inputs[i] if (index in inverse.keys())]
            words.append(word)
        return words
    
    def fit(self, 
            epochs = 1,
            batch_size = 64,
            X_train = None,
            y_train = None,
            X_test = None,
            y_test = None
            ):
        with tf.Session(graph=self.train_graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            iteration = 0
            print('Start training model.')
            for epoch in range(epochs):        
                for X_batch, y_batch in self.get_batches(X_train, y_train, batch_size):  #mini-batch implementation
                    feed = {self.inputs_: X_batch,
                            self.labels_: y_batch,
                            self.keep_prob: 0.8,
                            }
                    if (iteration%100 != 0):
                        sess.run(self.optimizer, feed_dict=feed)
                    else:
                        loss_train, acc_train, _= sess.run([self.loss, self.accuracy, self.optimizer], feed_dict=feed)
                        acc_test = []
                        for X_batch, y_batch in self.get_batches(X_test, y_test, 2000): #check accuracy in test set 
                            feed_test = {self.inputs_: X_batch,
                                         self.labels_: y_batch,
                                         self.keep_prob: 1,
                                        }
                            acc_test_batch = sess.run(self.accuracy, feed_dict=feed_test)
                            acc_test.append(acc_test_batch)
                        print("Epoch: {}/{}".format(epoch, epochs),
                              "Iteration: {}".format(iteration),
                              "Train loss: {:.6f}".format(loss_train),
                              "Train accuracy: {:.6f}".format(acc_train),
                              "Test accuracy: {:.6f}".format(np.mean(acc_test)))
                    iteration +=1
            print("Saving weights to checkpoints folder")
            self.saver.save(sess, "checkpoints/sentiment.ckpt")

    def predict(self, 
                batch_size = 1000,
                X_test = None,
                y_test = None,
                word_dict = None
                ):
        print('Make predictions and writes to file.')
        with open("results/prediction.txt","w+") as output:
            output.write('%5s %35s %35s \n' % ('Prediction', 'Top two features', 'Original review'))
            print('Start writing prediction file.')
            with tf.Session(graph=self.train_graph) as sess:
                self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
                batch_id = 1
                for X_batch, y_batch in self.get_batches(X_test, y_test, batch_size):
                    feed = {self.inputs_: X_batch,
                            self.labels_: y_batch,
                            self.keep_prob: 1.0
                            }
                    original_review = self.index_to_word(X_batch, word_dict)
                    attention_word = []
                    pred, feature_importance = sess.run([self.prediction, self.context_weights], feed_dict = feed)
                    attention_index = self.top_n_feature(X_batch, feature_importance, 2)
                    attention_word = self.index_to_word(attention_index, word_dict)
                    print('Writing batch %d. (%d reviews).' % (batch_id, len(pred)))
                    batch_id = batch_id + 1
                    for i in range(len(pred)):
                        label_pred = np.argmax(pred[i])
                        output.write('%5s %34s %34s \n' % (str(label_pred), \
                                                        ' & '.join(word for word in attention_word[i]),\
                                                            ''.join(word for word in original_review[i])))
                print('Finish writing prediction file.') 
        print('End.')

    