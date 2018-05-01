"""
Code idea from https://github.com/shekkizh/FCN.tensorflow
"""

import tensorflow as tf
import numpy as np
import fcn_model as fcn
import BatchDatasetReader as dataset
import ReadData as read

mode = "train"
data_dir = "/data"
#data_dir = "zzzkkkyyy/datasets/data/1"
saver_dir = "saver/"
check_dir = "checkpoint/"

adam_beta1 = 0.88
adam_lr = 5e-4

input_width = 256
input_height = 256
input_channels = 3
batch_size = 2


def main(argv = None):
    
    x = tf.placeholder(tf.float32, [None, input_width, input_height, input_channels])
    y = tf.placeholder(tf.int32, [None, input_width, input_height, 1])
    is_training = tf.placeholder(tf.bool)
    
    y_out, logits = fcn.construct_layer(x, is_training)
    tf.summary.image("input_image", x, max_outputs = 2)
    tf.summary.image("ground_truth", tf.cast(y, tf.uint8), max_outputs = 2)
    tf.summary.image("pred_annotation", tf.cast(y_out, tf.uint8), max_outputs = 2)
    #total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_out, labels = y)
    total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = tf.squeeze(y, squeeze_dims = [3]), name = "entropy")
    
    trainable_var = tf.trainable_variables()
    mean_loss = tf.reduce_mean(total_loss)
    tf.summary.scalar("entropy", mean_loss)
    optimizer = tf.train.AdamOptimizer(adam_lr, beta1 = adam_beta1)
    #train_op = optimizer.minimize(mean_loss)
    grads = optimizer.compute_gradients(mean_loss, var_list = trainable_var)
    train_op = optimizer.apply_gradients(grads)
    
    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
    print("Setting up image reader...")
    train_records, valid_records = read.read_dataset(data_dir)
    
    sess = tf.Session()
    
    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(saver_dir, sess.graph)
    
    print("Setting up dataset reader...")
    image_options = {'resize': True, 'resize_size': input_width}
    if mode == "train":
        train_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    else:
        validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    
    print("global variables initializing...")
    sess.run(tf.global_variables_initializer())
    print("global variables initialized!")
    ckpt = tf.train.get_checkpoint_state(check_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    
    if mode == "train":
        for itr in range(200):
            print("reading batches...")
            train_images, train_annotations = train_dataset_reader.next_batch(batch_size)
            print("feeding...")
            feed_dict = {x: train_images, y: train_annotations, is_training: True}
            print("run with feedings...")
            sess.run(train_op, feed_dict = feed_dict)
            if itr % 10 == 0:
                #train_loss = sess.run([total_loss], feed_dict = feed_dict)
                train_loss, summary_str = sess.run([total_loss, summary_op], feed_dict = feed_dict)
                print("Step: {}, Train_loss: {}".format(itr, train_loss))
                summary_writer.add_summary(summary_str, itr)
            """    
            if itr % 20 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(batch_size)
                valid_loss = sess.run(total_loss, feed_dict={x: valid_images, y: valid_annotations})
                print("Step: {} ---> Validation_loss: {}".format(itr, valid_loss))
                saver.save(sess, check_dir + "model.ckpt", itr)
            """

    elif mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(batch_size)
        pred = sess.run(y_out, feed_dict={x: valid_images, y: valid_annotations})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)
    
    
if __name__ == "__main__":
    tf.app.run()    
    
    
