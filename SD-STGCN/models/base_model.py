from models.layers import *
from os.path import join as pjoin
import tensorflow as tf


def build_model(x, y, n_frame, Ks, Kt, blocks, keep_prob, sconv):
    '''
    Build the base model.
    x: placeholder features, [-1, n_frame, n, n_channel]
    y: placeholder label, [-1, n]
    n_frame: int, size of records for training.
    Ks: int, kernel size of spatial convolution.
    Kt: int, kernel size of temporal convolution.
    blocks: list, channel configs of st_conv blocks.
    keep_prob: placeholder.
    sconv: type of spatio-convolution layer, cheb or gcn
    '''

    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_frame

    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, sconv, act_func='GLU')
        Ko -= 2 * (Kt - 1)

    # Output Layer
    if Ko > 1:
        # logits shape: [-1, n_node]
        logits = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')



    train_loss = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                            labels=y, logits=logits, axis=-1))


    y_pred = tf.nn.softmax(logits)
    tf.compat.v1.add_to_collection(name='y_pred', value=y_pred)

    return train_loss, y_pred

def build_model_SI(x, y, n_frame, Ks, Kt, blocks, keep_prob, sconv):
    '''
    Build the base model.
    x: placeholder features, [-1, n_frame, n, n_channel]
    y: placeholder label, [-1, n]
    n_frame: int, size of records for training.
    Ks: int, kernel size of spatial convolution.
    Kt: int, kernel size of temporal convolution.
    blocks: list, channel configs of st_conv blocks.
    keep_prob: placeholder.
    sconv: type of spatio-convolution layer, cheb or gcn
    '''

    Input = x
    batch_size = tf.shape(Input)[0]

    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_frame

    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, sconv, act_func='GLU')
        Ko -= 2 * (Kt - 1)

    # Output Layer
    if Ko > 1:
        # logits shape: [-1, n_node]
        logits = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    # ---- Apply SI propagation constraints ----
    # Extract infection status: shape = [batch_size, n_node]
    infection_status = Input[:, 0, :, 1]  # Select the second feature (index 1) at time step 0

    # Create mask for infected nodes
    infected_mask = tf.cast(infection_status > 0, tf.float32)  # 1 for infected, 0 for non-infected

    # Set logits to zero for non-infected nodes
    logits = logits * infected_mask + (1 - infected_mask) * (-1e9) # Zeroing out non-infected nodes

    # Compute loss
    train_loss = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                            labels=y, logits=logits, axis=-1))


    y_pred = tf.nn.softmax(logits)
    tf.compat.v1.add_to_collection(name='y_pred', value=y_pred)

    return train_loss, y_pred

def build_model_SI_nodewise(x, y, n_frame, Ks, Kt, blocks, keep_prob, sconv, pos_weight):
    '''
    Build the base model for node-wise binary classification (label vs not label).
    x: placeholder features, shape [-1, n_frame, n_node, n_channel]
    y: placeholder labels, shape [-1, n_node] (0 or 1 per node)
    n_frame: number of frames
    Ks: spatial kernel size
    Kt: temporal kernel size
    blocks: list of output channels for each ST-Conv block
    keep_prob: dropout keep probability
    sconv: type of spatial convolution ('gcn' or 'cheb')
    pos_weight: weight put on the postive samples. Due to the imbalance of labels.
    '''
    Input = x
    batch_size = tf.shape(Input)[0]

    # Initial Ko (for controlling temporal size)
    Ko = n_frame

    # ST-Block: spatial-temporal convolution
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, sconv, act_func='GLU')
        Ko -= 2 * (Kt - 1)

    # Output Layer: get logits with shape [-1, n_node, 2]
    if Ko > 1:
        logits = output_layer_nodewise(x, Ko, 'output_layer_nodewise')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    # ---- Apply SI propagation constraints ----
    # Infection status: select feature at time=0, feature=1
    infection_status = Input[:, 0, :, 1]  # shape: [batch_size, n_node]

    # Create infected mask
    infected_mask = tf.cast(infection_status > 0, tf.float32)  # [batch_size, n_node]

    # Expand and tile to match logits shape
    infected_mask = tf.expand_dims(infected_mask, axis=-1)     # [batch_size, n_node, 1]
    infected_mask = tf.tile(infected_mask, [1, 1, 2])           # [batch_size, n_node, 2]

    # Set logits to very negative for non-infected nodes (both classes)

    logits = logits * infected_mask + (1.0 - infected_mask) * (-1e9)

    # ---- Compute Weighted Loss ----

    # pos_weight > 1 means we care more about predicting label=1 correctly

    # Create class weights: 1 for label 0, pos_weight for label 1
    class_weights = tf.where(tf.equal(y, 1),
                            tf.ones_like(y, dtype=tf.float32) * pos_weight,
                            tf.ones_like(y, dtype=tf.float32))  # shape [batch_size, n_node]

    # Get per-node cross-entropy loss
    per_node_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)  # shape [batch_size, n_node]

    # Apply class weights
    weighted_loss = per_node_loss * class_weights  # shape [batch_size, n_node]

    # Final loss: mean over all nodes and batches
    train_loss = tf.reduce_mean(weighted_loss)

    y_pred = tf.nn.softmax(logits, axis=-1)  # shape [-1, n_node, 2]
    tf.compat.v1.add_to_collection(name='y_pred', value=y_pred)

    return train_loss, y_pred



def model_save(sess, global_steps, model_name, save_path='./output/models/'):
    '''
    Save the checkpoint of trained model.
    sess: tf.Session().
    global_steps: tensor, record the global step of training in epochs.
    model_name: str, the name of saved model.
    save_path: str, the path of saved model.
    '''
    saver = tf.compat.v1.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
