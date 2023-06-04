import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

plt.ion()

txt_path_train = r"C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\scratch_training\sleeptransformer\scratch_training_3chan_CPU_test2\n1\train_log.txt"
txt_path_evaluation = r"C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\scratch_training\sleeptransformer\scratch_training_3chan_CPU_test2\n1\eval_result_log.txt"


data_train = pd.read_csv(txt_path_train, sep=" ", header=None)
data_eval = pd.read_csv(txt_path_evaluation, sep=" ", header=None)

plt.figure(figsize=[7.4, 5.6])
plt.plot(data_train.iloc[:, 1], color="royalblue")
plt.plot(np.arange(len(data_eval))*3825, data_eval.iloc[:, 0]/560, '.-', markersize=14, linewidth=3, color="seagreen") # /560 because in the evaluation the loss is the sum of the loss across all batches, and there are 560 batches
plt.xlabel('Minibatches', fontsize=15)
plt.legend(['Training', 'Evaluation'], fontsize=12)
plt.ylabel('Cross entropy', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.figure(figsize=[7.4, 5.6])
plt.plot(data_train.iloc[:, 3])
plt.plot(np.arange(len(data_eval))*3825, data_eval.iloc[:, 2], '.-', markersize=9)
plt.xlabel('Minibatches', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(['Training', 'Evaluation'], fontsize=12)

plt.figure(figsize=[9, 6.8])
plt.plot(data_train.iloc[:, 1].rolling(window=15).mean(), color="royalblue", linewidth=1.75)
plt.plot(np.arange(len(data_eval))*3825, data_eval.iloc[:, 0]/560, '.-', markersize=17, linewidth=3.5, color="mediumseagreen")
plt.xlabel('Minibatches', fontsize=18)
plt.ylabel('Cross entropy', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.plot((np.arange(len(data_eval))*3825)[9], (data_eval.iloc[:, 0]/560)[9], '.', markersize=18, color='tomato')
plt.legend(['Training', 'Evaluation', 'Selected model'], fontsize=17)


plt.figure(figsize=[9, 6.8])
plt.plot(data_train.iloc[:, 3].rolling(window=15).mean(), color="royalblue", linewidth=1.75)
plt.plot(np.arange(len(data_eval))*3825, data_eval.iloc[:, 2], '.-', markersize=17, linewidth=3.5, color="mediumseagreen")
plt.xlabel('Minibatches', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.plot((np.arange(len(data_eval))*3825)[9], (data_eval.iloc[:, 2])[9], 'r.', markersize=18, color="tomato")
plt.legend(['Training', 'Evaluation', 'Selected model'], fontsize=17)

# d2=data.groupby(np.arange(len(data))//400).mean()
# plt.figure()
# plt.plot(d2.iloc[:, 1])
# plt.title('Training output loss')
# plt.figure()
# plt.plot(d2.iloc[:, 3])
# plt.title('Training accuracy')


#
# plt.figure()
# plt.plot(np.arange(len(data))*3825, data.iloc[:, 0], '.-', markersize=9)
# plt.title('Evaluation output loss', fontsize=15)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.xlabel('Minibatches', fontsize=15)
# plt.figure()
# plt.plot(np.arange(len(data))*3825, data.iloc[:, 2], '.-', markersize=9)
# plt.title('Evaluation accuracy', fontsize=15)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.xlabel('Minibatches', fontsize=15)
a=8


A1_acc_training = [.93, .944, .948, .951, .9535, .957, .96, .962, .965, .966]
A1_acc_val = [.94, .945, .952, .954, .952, .955, .959, .955, .957, .958]
A1_loss_training = [.208, .163, .148, .138, .124, .115, .105, .096, .088, .08]
A1_loss_val = [.164, .144, .143, .14, .144, .138, .149, .14, .153, .17]

plt.figure(figsize=[9, 6.8])
plt.plot(np.arange(10)+1, A1_acc_training, '.-', markersize=17,  linewidth=3.5, color="royalblue")
plt.plot(np.arange(10)+1, A1_acc_val, '.-', markersize=17, linewidth=3.5, color="mediumseagreen")
plt.plot((np.arange(10)+1)[4], A1_acc_training[4], '.', markersize=18, color="tomato")
plt.plot((np.arange(10)+1)[4], A1_acc_val[4], '.', markersize=18, color="tomato")
plt.xlabel('Epochs', fontsize=18)
plt.legend(['Training', 'Evaluation', 'Selected model'], fontsize=17)
plt.ylabel('Accuracy', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.ylim([0.84, 1.00001])

plt.figure(figsize=[9, 6.8])
plt.plot((np.arange(10)+1), A1_loss_training, '.-', markersize=17, linewidth=3.5, color="royalblue")
plt.plot((np.arange(10)+1), A1_loss_val, '.-', markersize=17, linewidth=3.5, color="mediumseagreen")
plt.plot((np.arange(10)+1)[4], A1_loss_training[4], '.', markersize=18, color="tomato")
plt.plot((np.arange(10)+1)[4], A1_loss_val[4], '.', markersize=18, color="tomato")
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Class weighted CE', fontsize=18)
# plt.yticks(np.arange(0, 1.1, 0.1))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.legend(['Training', 'Evaluation', 'Selected model'], fontsize=17)


# fig, ax = plt.subplots(3, 1, figsize=(5, 12))
# ax[0].plot(A1_acc_training)
# ax[0].plot(A1_acc_val)
# ax[1].plot(A1_loss_training)
# ax[1].plot(A1_loss_val)
# # plt.plot(np.arange(len(data_eval))*3825, data_eval.iloc[:, 2], '.-', markersize=14, linewidth=2)
# plt.xlabel('Minibatches', fontsize=18)
# ax[0].set_ylim([0.92, .975])
# ax[1].set_ylim([0.08, .21])
# plt.ylabel('Accuracy', fontsize=18)
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.plot((np.arange(len(data_eval))*3825)[9], (data_eval.iloc[:, 2])[9], 'r.', markersize=14)
# plt.legend(['Training', 'Evaluation', 'Best model'], fontsize=17)

B1_acc_training = [0.94, 0.97, 0.925, .97, .975, .98, 0.9825, 0.97, 0.985, 0.97]
B1_acc_val = [.91, .9375, .945, .955, .965, .9675, .969, .97, .973, .974]
B1_loss_training = [0.45, 0.275, 0.225, .2175, .175, .15, 0.125, 0.13, 0.1, 0.15]
B1_loss_val = [.225, .205, .205, .17, .1685, .1665, .18, .1675, .182, .1665]


plt.figure(figsize=[9, 6.8])
plt.plot(np.arange(10)+1, B1_acc_training, '.-', markersize=17,  linewidth=3.5, color="royalblue")
plt.plot(np.arange(10)+1, B1_acc_val, '.-', markersize=17,  linewidth=3.5, color="mediumseagreen")
plt.plot((np.arange(10)+1)[4], B1_acc_training[4], 'r.', markersize=18, color="tomato")
plt.plot((np.arange(10)+1)[4], B1_acc_val[4], 'r.', markersize=18, color="tomato")
plt.xlabel('Epochs', fontsize=18)
plt.legend(['Training', 'Evaluation', 'Selected model'], fontsize=17)
plt.ylabel('Accuracy', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.ylim([0.85, 1.00001])

plt.figure(figsize=[9, 6.8])
plt.plot(np.arange(10)+1, B1_loss_training, '.-', markersize=17,  linewidth=3.5, color="royalblue")
plt.plot(np.arange(10)+1, B1_loss_val, '.-', markersize=17,  linewidth=3.5, color="mediumseagreen")
plt.plot((np.arange(10)+1)[4], B1_loss_training[4], 'r.', markersize=18, color="tomato")
plt.plot((np.arange(10)+1)[4], B1_loss_val[4], 'r.', markersize=18, color="tomato")
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Class weighted CE', fontsize=18)
# plt.yticks(np.arange(0, 1.1, 0.1))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.legend(['Training', 'Evaluation', 'Selected model'], fontsize=17)






plt.figure(figsize=[7.4, 5.6])
plt.plot(np.arange(10)+1, A1_acc_training, '.-', markersize=9)
plt.plot(np.arange(10)+1, A1_acc_val, '.-', markersize=9)
plt.plot((np.arange(10)+1)[4], A1_acc_training[4], 'r.', markersize=9)
plt.plot((np.arange(10)+1)[4], A1_acc_val[4], 'r.', markersize=9)
plt.plot(np.arange(10)+1, B1_acc_training, '.-', markersize=9)
plt.plot(np.arange(10)+1, B1_acc_val, '.-', markersize=9)
plt.plot((np.arange(10)+1)[4], B1_acc_training[4], 'r.', markersize=9)
plt.plot((np.arange(10)+1)[4], B1_acc_val[4], 'r.', markersize=9)
plt.xlabel('Epochs', fontsize=15)
plt.legend(['Training CNN1', 'Evaluation CNN1', 'Training CNN2', 'Evaluation CNN2', 'Selected model'], fontsize=12)
plt.ylabel('Accuracy', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylim([0.85, 1.00001])

plt.figure(figsize=[7.4, 5.6])
plt.plot((np.arange(10)+1), A1_loss_training, '.-', markersize=9)
plt.plot((np.arange(10)+1), A1_loss_val, '.-', markersize=9)
plt.plot((np.arange(10)+1)[4], A1_loss_training[4], 'r.', markersize=9)
plt.plot((np.arange(10)+1)[4], A1_loss_val[4], 'r.', markersize=9)
plt.plot(np.arange(10)+1, B1_loss_training, '.-', markersize=9)
plt.plot(np.arange(10)+1, B1_loss_val, '.-', markersize=9)
plt.plot((np.arange(10)+1)[4], B1_loss_training[4], 'r.', markersize=9)
plt.plot((np.arange(10)+1)[4], B1_loss_val[4], 'r.', markersize=9)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Class weighted CE', fontsize=15)
# plt.yticks(np.arange(0, 1.1, 0.1))
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(['Training CNN1', 'Evaluation CNN1', 'Training CNN2', 'Evaluation CNN2', 'Selected model'], fontsize=12)


# fig, ax = plt.subplots(3, 1, figsize=(5, 12))
# ax[0].plot(B1_acc_training)
# ax[0].plot(B1_acc_val)
# ax[1].plot(B1_loss_training)
# ax[1].plot(B1_loss_val)
# # plt.plot(np.arange(len(data_eval))*3825, data_eval.iloc[:, 2], '.-', markersize=14, linewidth=2)
# plt.xlabel('Minibatches', fontsize=18)
# ax[0].set_ylim([0.7, .99])
# ax[1].set_ylim([0.08, .45])
# plt.ylabel('Accuracy', fontsize=18)
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.plot((np.arange(len(data_eval))*3825)[9], (data_eval.iloc[:, 2])[9], 'r.', markersize=14)
# plt.legend(['Training', 'Evaluation', 'Best model'], fontsize=17)





