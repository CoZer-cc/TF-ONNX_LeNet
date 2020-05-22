import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images
X_test = mnist.test.images
Y_train = mnist.train.labels
Y_test = mnist.test.labels

n_classes = 10 # 0-9 digits
n_width = 28
n_height = 28
n_depth = 1
n_inputs = n_height * n_width * n_depth # total pixels
learning_rate = 0.001
n_epochs = 10
batch_size = 100
n_batches = int(mnist.train.num_examples/batch_size)

# input images shape: (n_samples,n_pixels)
x = tf.placeholder(dtype=tf.float32, name="x", shape=[None, n_inputs]) 
# output labels
y = tf.placeholder(dtype=tf.float32, name="y", shape=[None, n_classes])

x_ = tf.reshape(x, shape=[-1, n_width, n_height, n_depth])


layer1_w = tf.Variable(tf.random_normal(shape=[4,4,n_depth,32],
            stddev=0.1),name='l1_w')
layer1_b = tf.Variable(tf.random_normal([32]),name='l1_b')

layer1_conv = tf.nn.relu(tf.nn.conv2d(x_,layer1_w,
                                      strides=[1,1,1,1],
                                      padding='SAME'
                                     ) + 
                         layer1_b 
                        )

layer1_pool = tf.nn.max_pool(layer1_conv,ksize=[1,2,2,1],
                 strides=[1,2,2,1],padding='SAME')

layer2_w = tf.Variable(tf.random_normal(shape=[4,4,32,64],
           stddev=0.1),name='l2_w')
layer2_b = tf.Variable(tf.random_normal([64]),name='l2_b')

layer2_conv = tf.nn.relu(tf.nn.conv2d(layer1_pool,
                                      layer2_w,
                                      strides=[1,1,1,1],
                                      padding='SAME'
                                     ) + 
                          layer2_b
                        )

layer2_pool = tf.nn.max_pool(layer2_conv,
                             ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME'
                            )

layer3_w = tf.Variable(tf.random_normal(shape=[64*7*7*1,1024],
                       stddev=0.1),name='l3_w')
layer3_b = tf.Variable(tf.random_normal([1024]),name='l3_b')
layer3_fc = tf.nn.relu(tf.matmul(tf.reshape(layer2_pool,
            [-1, 64*7*7*1]),layer3_w) + layer3_b)

layer4_w = tf.Variable(tf.random_normal(shape=[1024, n_classes],
                                        stddev=0.1),name='l')

layer4_b = tf.Variable(tf.random_normal([n_classes]),name='l4_b')
layer4_out = tf.matmul(layer3_fc,layer4_w)+layer4_b
pred = tf.nn.softmax(layer4_out, name='out_softmax')
model = layer4_out

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y,name='output')
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)




with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in range(n_batches):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            feed_dict={x:batch_x, y: batch_y}
            batch_loss,_ = sess.run([loss, optimizer],
                                   feed_dict=feed_dict)
            total_loss += batch_loss 
        average_loss = total_loss / n_batches
        print("Epoch: {0:04d} loss = {1:0.6f}".format(epoch,average_loss))
    tf.saved_model.simple_save(
        sess,
        "./savedmodel/",
        inputs={"x":x},
        outputs={"output":entropy})
    

    predictions_check = tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(predictions_check, tf.float32))
    feed_dict = {x:mnist.test.images, y:mnist.test.labels}
    print("Accuracy:", accuracy.eval(feed_dict=feed_dict))
 # 保存模型
"""         constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["out_softmax"])  # out_softmax
        with tf.gfile.FastGFile("model.pb", mode='wb') as f:
            f.write(constant_graph.SerializeToString()) 
            
            print("Model Trained.")
    tf.saved_model.simple_save(
        sess,
        "./savedmodel/",
        inputs={"x":x},
        outputs={"y":y}"""