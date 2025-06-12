from data_loader.data_utils import *
from utils.metric_utils import *
# from models.base_model import build_model, build_model_SI, build_model_SI_nodewise, model_save
from os.path import join as pjoin
from models.base_model import STGCN_SI_Nodewise, model_save
# import tensorflow as tf
import numpy as np
import time
import os
import shutil
import torch
import torch.nn.functional as F

# def model_train(inputs, blocks, args, save_path='./output/models/', sum_path='./output/tensorboard'):
#     '''
#     Train the base model.
#     inputs: instance of class Dataset, data source for training.
#     blocks: list, channel configs of st_conv blocks.
#     args: instance of class argparse, args for training.
#     '''
#     n, n_frame, n_channel = args.n_node, args.n_frame, args.n_channel
#     Ks, Kt = args.ks, args.kt
#     batch_size, epoch, opt = args.batch_size, args.epoch, args.opt

#     sconv = args.sconv # spatio-conv type

#     dropout = args.dropout

#     start, end = args.start, args.end

#     valid = args.valid

#     random = args.random

#     # Placeholder for model training
#     x = tf.compat.v1.placeholder(tf.float32, [None, n_frame, n, n_channel], name='data_input')

#     # placeholder for one-hot labels
#     y = tf.compat.v1.placeholder(tf.float32, [None, n], name='data_label')

#     keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

#     # Define model loss
#     train_loss, pred = build_model_SI(x, y, n_frame, Ks, Kt, blocks, keep_prob, sconv)
#     tf.compat.v1.summary.scalar('train_loss', train_loss)

#     # Learning rate settings
#     global_steps = tf.Variable(0, trainable=False)
#     len_train = inputs.get_len('train')

#     if len_train % batch_size == 0:
#         epoch_step = len_train / batch_size
#     else:
#         epoch_step = int(len_train / batch_size) + 1

#     # Learning rate decay with rate 0.7 every 5 epochs.
#     lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
#     tf.compat.v1.summary.scalar('learning_rate', lr)
#     step_op = tf.compat.v1.assign_add(global_steps, 1)
#     with tf.control_dependencies([step_op]):
#         if opt == 'RMSProp':
#             train_op = tf.compat.v1.train.RMSPropOptimizer(lr).minimize(train_loss)
#         elif opt == 'ADAM':
#             train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(train_loss)
#         else:
#             raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

#     merged = tf.compat.v1.summary.merge_all()

#     # Initialize variables to track the best validation accuracy and corresponding model
#     best_val_acc = 0.0
#     best_epoch = 0

#     with tf.compat.v1.Session() as sess:
#         sess.run(tf.compat.v1.global_variables_initializer())

#         for i in range(epoch):
#             start_time = time.time()
#             for j, (x_batch, y_batch) in enumerate(
#                 gen_xy_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):

#                 x_batch_ = onehot(iteration2snapshot(x_batch, n_frame, start=start, end=end, random=random), n_channel) 
#                 summary, _ = sess.run([merged, train_op],\
#                                 feed_dict={x: x_batch_,\
#                                 y: onehot(y_batch, n), keep_prob: 1.0})


#                 if j % 10 == 0:
#                     loss_value, pred_batch = \
#                         sess.run([train_loss,pred],
#                                  feed_dict={x: x_batch_, y: onehot(y_batch,n), keep_prob: 1-dropout})

#                     acc = batch_acc(pred_batch, y_batch)

#                     if valid:
#                         # evaluate acc on validation
#                         acc_val_list = []
#                         for (x_val, y_val) in gen_xy_batch(inputs.get_data('val'), batch_size, dynamic_batch=True, shuffle=False):

#                             x_val_ = onehot(iteration2snapshot(x_val,n_frame,start=start,end=end,random=random), n_channel)


#                             pred_val = sess.run(pred, feed_dict={x: x_val_, y:\
#                                                         onehot(y_val, n), keep_prob: 1.0})
#                             acc_val_list.append(batch_acc(pred_val, y_val))


#                         acc_val = np.mean(acc_val_list)
#                         end_time = time.time()
#                         print(f'Epoch {i:2d}, Step {j:3d}: Elapsed Time {(end_time-start_time):.3f} Loss {loss_value:.3f} Train Acc {acc:.3f} Val Acc {acc_val:.3f}')
#                         # Check if the current validation accuracy is the best
#                         if acc_val > best_val_acc:
#                             best_val_acc = acc_val
#                             best_epoch = i
#                             # delete the previous best model
#                             if os.path.exists(save_path):
#                                 # delete all files in save_path with name starting by STGCN
#                                 for file in os.listdir(save_path):
#                                     if file.startswith('STGCN'):
#                                         os.remove(os.path.join(save_path, file))
#                             # Save the model when a new best validation accuracy is found
#                             model_save(sess, global_steps, 'STGCN', save_path)
#                             print(f'New best model saved with validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}')
#                     else:
#                         end_time = time.time()
#                         print(f'Epoch {i:2d}, Step {j:3d}: Elapsed Time {(end_time-start_time):.3f} Loss {loss_value:.3f} Train Acc {acc:.3f}')

#         # Save the final model only if validation is enabled and it's the best
#         if valid:
#             print(f'Final model saved with best validation accuracy: {best_val_acc:.3f}')
#         else:
#             # Save the final model directly if validation is not enabled
#             model_save(sess, global_steps, 'STGCN', save_path)
#             print('Final model saved.')

#             # if (i + 1) % args.save == 0:
#             #     model_save(sess, global_steps, 'STGCN', save_path)
#     print('Training model finished!')

# def model_train_nodewise(inputs, blocks, args, save_path='./output/models/', sum_path='./output/tensorboard'):
#     '''
#     Train the base model for node-wise classification.
#     '''
#     n, n_frame, n_channel = args.n_node, args.n_frame, args.n_channel
#     Ks, Kt = args.ks, args.kt
#     batch_size, epoch, opt = args.batch_size, args.epoch, args.opt

#     sconv = args.sconv
#     pos_weight = args.pos_weight
#     dropout = args.dropout
#     start, end = args.start, args.end
#     valid = args.valid
#     random = args.random

#     prop_model = args.prop_model

#     # Placeholders
#     x = tf.compat.v1.placeholder(tf.float32, [None, n_frame, n, n_channel], name='data_input')
#     y = tf.compat.v1.placeholder(tf.int32, [None, n], name='data_label')  # 🔥 Labels are int (0/1), NOT one-hot
#     keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

#     # Build the model
#     train_loss, pred = build_model_SI_nodewise(x, y, n_frame, Ks, Kt, blocks, keep_prob, sconv, pos_weight, prop_model)
#     tf.compat.v1.summary.scalar('train_loss', train_loss)

#     # Learning rate and optimizer
#     global_steps = tf.Variable(0, trainable=False)
#     len_train = inputs.get_len('train')

#     epoch_step = len_train // batch_size + int(len_train % batch_size != 0)

#     lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
#     tf.compat.v1.summary.scalar('learning_rate', lr)
    
#     step_op = tf.compat.v1.assign_add(global_steps, 1)
#     with tf.control_dependencies([step_op]):
#         if opt == 'RMSProp':
#             train_op = tf.compat.v1.train.RMSPropOptimizer(lr).minimize(train_loss)
#         elif opt == 'ADAM':
#             train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(train_loss)
#         else:
#             raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

#     merged = tf.compat.v1.summary.merge_all()

#     best_val_acc = 0.0
#     best_epoch = 0

#     with tf.compat.v1.Session() as sess:
#         sess.run(tf.compat.v1.global_variables_initializer())

#         for i in range(epoch):
#             start_time = time.time()

#             for j, (x_batch, y_batch) in enumerate(
#                 gen_xy_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):

#                 x_batch_ = onehot(iteration2snapshot(x_batch, n_frame, start=start, end=end, random=random), n_channel)

#                 y_batch = snapshot_to_labels(y_batch, n)
                
#                 # 🔥 No one-hot on labels anymore
#                 summary, _ = sess.run([merged, train_op],
#                                       feed_dict={x: x_batch_, y: y_batch, keep_prob: 1.0})

#                 if j % 10 == 0:
#                     loss_value, pred_batch = sess.run(
#                         [train_loss, pred],
#                         feed_dict={x: x_batch_, y: y_batch, keep_prob: 1 - dropout})

#                     acc = batch_acc_nodewise(pred_batch, y_batch)  # node-wise accuracy

#                     if valid:
#                         acc_val_list = []
#                         for (x_val, y_val) in gen_xy_batch(inputs.get_data('val'), batch_size, dynamic_batch=True, shuffle=False):
#                             x_val_ = onehot(iteration2snapshot(x_val, n_frame, start=start, end=end, random=random), n_channel)
#                             y_val_ = snapshot_to_labels(y_val, n)
                            
#                             pred_val = sess.run(pred, feed_dict={x: x_val_, y: y_val_, keep_prob: 1.0})
#                             acc_val_list.append(batch_acc_nodewise(pred_val, y_val_))

#                         acc_val = np.mean(acc_val_list)
#                         end_time = time.time()
#                         print(f'Epoch {i:2d}, Step {j:3d}: Elapsed Time {(end_time-start_time):.3f} Loss {loss_value:.3f} Train Acc {acc:.3f} Val Acc {acc_val:.3f}')
                        
#                         if acc_val > best_val_acc:
#                             best_val_acc = acc_val
#                             best_epoch = i
#                             if os.path.exists(save_path):
#                                 for file in os.listdir(save_path):
#                                     if file.startswith('STGCN'):
#                                         os.remove(os.path.join(save_path, file))
#                             model_save(sess, global_steps, 'STGCN', save_path)
#                             print(f'New best model saved with validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}')
#                     else:
#                         end_time = time.time()
#                         print(f'Epoch {i:2d}, Step {j:3d}: Elapsed Time {(end_time-start_time):.3f} Loss {loss_value:.3f} Train Acc {acc:.3f}')

#         if valid:
#             print(f'Final model saved with best validation accuracy: {best_val_acc:.3f}')
#         else:
#             model_save(sess, global_steps, 'STGCN', save_path)
#             print('Final model saved.')

#     print('Training model finished!')


def model_train_pytorch_nodewise(inputs, blocks, args, save_path='./output/models/'):
    '''
    Train the base model for node-wise classification using PyTorch.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n, n_frame, n_channel = args.n_node, args.n_frame, args.n_channel
    Ks, Kt = args.ks, args.kt
    batch_size, epochs, opt = args.batch_size, args.epoch, args.opt
    sconv = args.sconv
    pos_weight = args.pos_weight
    dropout = args.dropout
    start, end = args.start, args.end
    valid = args.valid
    random = args.random
    prop_model = args.prop_model
    Lk = inputs.Lk  # graph Laplacian kernels (assumed torch.Tensor list)

    model = STGCN_SI_Nodewise(n_frame, Ks, Kt, blocks, sconv, Lk, 1 - dropout, pos_weight).to(device)

    if opt == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif opt == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    # decay_steps = 500
    # decay_rate = 0.7
 
    # lr_lambda = lambda step: decay_rate ** (step // decay_steps)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        for j, (x_batch, y_batch) in enumerate(gen_xy_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
            x_batch_np = onehot(iteration2snapshot(x_batch, n_frame, start, end, random), n_channel)
            y_batch_np = snapshot_to_labels(y_batch, n)

            x_tensor = torch.tensor(x_batch_np, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_batch_np, dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits, loss = model(x_tensor, y_tensor, training=True)
            if j % 10 == 0:
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=-1)  # [B, N]
                    label = y_tensor  # [B, N]

                    correct = (pred == label).float()
                    acc = correct.sum() / correct.numel()

                    pred_flat = pred.view(-1).cpu().numpy()
                    label_flat = label.view(-1).cpu().numpy()

                    pred_count = np.bincount(pred_flat, minlength=2)
                    label_count = np.bincount(label_flat, minlength=2)

                    print(f"[Train Step {j}] Acc: {acc.item():.4f} | Pred: 0={pred_count[0]}, 1={pred_count[1]} | Label: 0={label_count[0]}, 1={label_count[1]}")

            loss.backward()
            optimizer.step()

            if j % 10 == 0:
                with torch.no_grad():
                    model.eval()
                    acc = batch_acc_nodewise(F.softmax(logits, dim=-1).cpu().numpy(), y_batch_np)

                    if valid:
                        acc_val_list = []
                        for (x_val, y_val) in gen_xy_batch(inputs.get_data('val'), batch_size, dynamic_batch=True, shuffle=False):
                            x_val_np = onehot(iteration2snapshot(x_val, n_frame, start, end, random), n_channel)
                            y_val_np = snapshot_to_labels(y_val, n)

                            x_val_tensor = torch.tensor(x_val_np, dtype=torch.float32).to(device)
                            y_val_tensor = torch.tensor(y_val_np, dtype=torch.long).to(device)

                            with torch.no_grad():
                                pred_val = model(x_val_tensor, training=False)
                                pred_val_softmax = F.softmax(pred_val, dim=-1).cpu().numpy()

                            acc_val_list.append(batch_acc_nodewise(pred_val_softmax, y_val_np))

                        acc_val = np.mean(acc_val_list)
                        end_time = time.time()
                        print(f'Epoch {epoch:2d}, Step {j:3d}: Time {end_time-start_time:.2f}s | Loss {loss.item():.3f} | Train F1 {acc:.3f} | Val F1 {acc_val:.3f}')

                        if acc_val > best_val_acc:
                            best_val_acc = acc_val
                            best_epoch = epoch
                            model_save(model, epoch, 'STGCN', save_path)
                            print(f'New best model saved with validation F1: {best_val_acc:.3f} at epoch {best_epoch}')
                    else:
                        end_time = time.time()
                        print(f'Epoch {epoch:2d}, Step {j:3d}: Time {end_time-start_time:.2f}s | Loss {loss.item():.3f} | Train F1 {acc:.3f}')

        scheduler.step()

    if not valid:
        model_save(model, epoch, 'STGCN', save_path)
        print('Final model saved.')
    else:
        print(f'Final model saved with best validation accuracy: {best_val_acc:.3f}')