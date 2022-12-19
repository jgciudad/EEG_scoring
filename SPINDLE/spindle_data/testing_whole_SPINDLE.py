from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from spindle_data.spindle_data_loading import load_to_dataset
from metrics import *
from tools import *
from hmm import *

saved_model_folder = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE_pycharm\results\3 - new round of results after meeting'
# save_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE\results\3 - new round of results after meeting\A_1\Evaluation\Against scorers intersection\Excluding artifacts\Before HMM'

plt.ion()

# -------------------------------------------------------------------------------------------------------------------------

test_dataset_3 = load_to_dataset(signal_paths=[r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A3.edf"],
                               labels_paths=[r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A3.csv'],
                               scorer=2,
                               just_artifact_labels=False,
                               artifact_to_stages=True,
                               balance_artifacts=False,
                               validation_split=0)

test_dataset_4 = load_to_dataset(signal_paths=[r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A4.edf"],
                               labels_paths=[r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A4.csv'],
                               scorer=2,
                               just_artifact_labels=False,
                               artifact_to_stages=True,
                               balance_artifacts=False,
                               validation_split=0)

batch_size = 100
test_dataset_3 = test_dataset_3.batch(batch_size)
test_dataset_4 = test_dataset_4.batch(batch_size)

# -------------------------------------------------------------------------------------------------------------------------

# test_sequence = SequenceDataset(r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed\d2\testing\Cohort A',
#                                 batch_size=100,
#                                 binary=False)

# -------------------------------------------------------------------------------------------------------------------------
transition_matrix = get_transition_matrix([r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\scorings\A1.csv",
                                           r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\scorings\A2.csv"],
                                           cancel_forbidden_transitions=True)

# initial_probs = np.array([1/3, 1/3, 1/3])
initial_probs = get_priors([r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\scorings\A1.csv",
                          r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\scorings\A2.csv"])

spindle_model_1 = tf.keras.Sequential([
    Input((160, 48, 3)),
    MaxPool2D(pool_size=(2, 3), strides=(2, 3)),
    Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    Dense(units=3, activation='softmax', kernel_initializer='glorot_uniform')
])

spindle_model_1.load_weights(os.path.join(saved_model_folder, 'A_1', 'A_1' + "_5epochs" + ".h5"))

spindle_model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss=MulticlassWeightedCrossEntropy_2(n_classes=3),
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               BinaryBalancedAccuracy(),
                               BinaryF1Score()])

spindle_model_2 = tf.keras.Sequential([
    Input((160, 48, 3)),
    MaxPool2D(pool_size=(2, 3), strides=(2, 3)),
    Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform')
])

spindle_model_2.load_weights(os.path.join(saved_model_folder, 'B_1', 'B_1' + "_5epochs" + ".h5"))

spindle_model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss=BinaryWeightedCrossEntropy,
                      # BinaryWeightedCrossEntropy tf.keras.losses.BinaryCrossentropy()
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               BinaryBalancedAccuracy(),
                               BinaryF1Score()])


for idx, batch in enumerate(test_dataset_3):
    if idx == 0:
        y_pred_3_cnn1 = spindle_model_1(batch[0])
        y_pred_3_cnn2 = spindle_model_2(batch[0])
        y_true_3 = batch[1]
    else:
        y_pred_3_cnn1 = tf.concat([y_pred_3_cnn1, spindle_model_1(batch[0])], axis=0)
        y_pred_3_cnn2 = tf.concat([y_pred_3_cnn2, spindle_model_2(batch[0])], axis=0)
        y_true_3 = tf.concat([y_true_3, batch[1]], axis=0)

for idx, batch in enumerate(test_dataset_4):
    if idx == 0:
        y_pred_4_cnn1 = spindle_model_1(batch[0])
        y_pred_4_cnn2 = spindle_model_2(batch[0])
        y_true_4 = batch[1]
    else:
        y_pred_4_cnn1 = tf.concat([y_pred_4_cnn1, spindle_model_1(batch[0])], axis=0)
        y_pred_4_cnn2 = tf.concat([y_pred_4_cnn2, spindle_model_2(batch[0])], axis=0)
        y_true_4 = tf.concat([y_true_4, batch[1]], axis=0)

# for idx in range(len(test_sequence)):
#     batch = test_sequence[idx]
#     if idx == 0:
#         y_pred = spindle_model(batch[0])
#         y_true = batch[1]
#     else:
#         y_pred = tf.concat([y_pred, spindle_model(batch[0])], axis=0)
#         y_true = tf.concat([y_true, batch[1]], axis=0)


y_true_3 = np.argmax(y_true_3, axis=1)
y_true_4 = np.argmax(y_true_4, axis=1)

y_pred_hmm_3, _, _ = viterbi(y_pred_3_cnn1, transition_matrix, initial_probs)
y_pred_hmm_4, _, _ = viterbi(y_pred_4_cnn1, transition_matrix, initial_probs)

y_pred_hmm_filtered_3 = y_pred_hmm_3[(y_pred_3_cnn2 < 0.5)[:, 0]]
y_pred_hmm_filtered_4 = y_pred_hmm_4[(y_pred_4_cnn2 < 0.5)[:, 0]]
y_true_3 = y_true_3[(y_pred_3_cnn2 < 0.5)[:, 0]]
y_true_4 = y_true_4[(y_pred_4_cnn2 < 0.5)[:, 0]]

save_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE\results\3 - new round of results after meeting\whole_model\SPINDLE data\scorer 2\stages (excluding artifacts)'
compute_and_save_metrics_cnn1(np.hstack((y_true_3, y_true_4)), np.hstack((y_pred_hmm_filtered_3, y_pred_hmm_filtered_4)), save_path, 'whole_model')






# n_corrected_epochs_3, correction_matrix_3, cnn_transitions_3, hmm_transitions_3 = evaluate_hmm_effect(y_true_3, y_pred_cnn_3, y_pred_hmm_3)
# n_corrected_epochs_4, correction_matrix_4, cnn_transitions_4, hmm_transitions_4 = evaluate_hmm_effect(y_true_4, y_pred_cnn_4, y_pred_hmm_4)

save_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE\results\3 - new round of results after meeting\whole_model\SPINDLE data\scorer 2\artifacts'
# plot_and_save_hmm_effect(n_corrected_epochs_3 + n_corrected_epochs_4,
#                          correction_matrix_3 + correction_matrix_4,
#                          cnn_transitions_3 + cnn_transitions_4,
#                          hmm_transitions_3 + hmm_transitions_4,
#                          save_path, model_name)

# CONCATENATE
y_true_3_artifacts = load_labels(labels_path=r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A3.csv',
                                 scorer=2,
                                 just_artifact_labels=True,
                                 artifact_to_stages=False)
y_true_4_artifacts = load_labels(labels_path=r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A4.csv',
                                 scorer=2,
                                 just_artifact_labels=True,
                                 artifact_to_stages=False)


compute_and_save_metrics_cnn2(np.hstack((y_true_3_artifacts, y_true_4_artifacts)), np.vstack((y_pred_3_cnn2, y_pred_4_cnn2)), 0.5, save_path, 'whole_model')







############################################ REMOVED FUNCTIONS ####################################################

#
# class MulticlassWeightedCrossEntropy_2(tf.keras.losses.Loss):
#     '''
#     Same implementation as in https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
#
#     Weights are calculated as w_i = n_samples / (n_classes * n_elements_class_i)
#     '''
#
#     def __init__(self, n_classes, name="class_weighted_cross_entropy"):
#         super().__init__(name=name)
#         self.n_classes = n_classes
#
#     def call(self, y_true, y_pred):
#
#         ce = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
#
#         for i in range(self.n_classes):
#             num_class_i = tf.shape(tf.where(y_true[:, i] == 1))[0]
#             # tf.print('num_class_i: ', num_class_i)
#
#             if num_class_i > 0:
#                 w = tf.shape(y_true)[0] / (self.n_classes * num_class_i)
#                 w = tf.cast(w, dtype=tf.float32)
#                 # tf.print('weight: ', w)
#
#                 ce = tf.tensor_scatter_nd_update(ce, tf.where(y_true[:, i] == 1), w * ce[y_true[:, i] == 1])
#                 # tf.print('ce updated')
#
#         return ce

# class MulticlassBalancedAccuracy(tf.keras.metrics.Metric):
#     '''
#     Gives the average of the recall over each batch, where the recall of each batch is the average of the recall of each
#     class in that batch.
#
#     Same implementation as in https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/metrics/_classification.py#L1933
#
#     I tested it and gives the same result as sklearn.metrics.balanced_accuracy_score(y_true, y_pred).
#     Code to test it:
#         y_true1 = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
#         y_pred1 = np.array([0, 0, 1, 0, 1, 0, 1, 2, 0])
#         y_true2 = np.array([0, 2, 0, 0, 0, 0, 1, 2, 2])
#         y_pred2 = np.array([0, 0, 1, 0, 1, 0, 1, 2, 0])
#         av_bacc = (sklearn.metrics.balanced_accuracy_score(y_true1, y_pred1) + sklearn.metrics.balanced_accuracy_score(y_true2, y_pred2))/2
#         print(av_bacc)
#         y_true1 = np.array(
#             [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
#         y_pred1 = np.array(
#             [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
#         y_true2 = np.array(
#             [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
#         y_pred2 = np.array(
#             [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
#         y_true1 = tf.convert_to_tensor(y_true1)
#         y_pred1 = tf.convert_to_tensor(y_pred1)
#         y_true2 = tf.convert_to_tensor(y_true2)
#         y_pred2 = tf.convert_to_tensor(y_pred2)
#         MBA = MulticlassBalancedAccuracy(n_classes=3)
#         MBA.update_state(y_true1,y_pred1)
#         MBA.update_state(y_true2,y_pred2)
#         print(MBA.result())
#     '''
#
#     def __init__(self, n_classes, name='multiclass_balanced_accuracy', **kwargs):
#         super(MulticlassBalancedAccuracy, self).__init__(name=name, **kwargs)
#         self.n_classes = n_classes
#         self.TP = tf.keras.metrics.TruePositives()
#         self.FN = tf.keras.metrics.FalseNegatives()
#         self.epoch_average_recall = tf.keras.metrics.Mean()
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#
#         batch_recalls_sum = tf.convert_to_tensor(0, dtype=tf.float32)
#
#         for i in range(self.n_classes):
#             if tf.shape(tf.where(y_true[:, i] == 1))[0] > 0:
#                 self.TP.reset_state()
#                 self.FN.reset_state()
#
#                 self.TP.update_state(y_true[y_true[:, i] == 1], y_pred[y_true[:, i] == 1])
#                 self.FN.update_state(y_true[y_true[:, i] == 1], y_pred[y_true[:, i] == 1])
#
#                 class_recall = self.TP.result() / (self.TP.result() + self.FN.result())
#
#                 batch_recalls_sum = batch_recalls_sum + class_recall
#
#                 # tf.print('class_recall: ', class_recall)
#
#         batch_average_recall = batch_recalls_sum / tf.math.count_nonzero(tf.reduce_sum(y_true, axis=0), dtype=tf.float32)
#
#         self.epoch_average_recall.update_state(batch_average_recall)
#
#     def result(self):
#         return self.epoch_average_recall.result()
#
#     def reset_state(self):
#         self.epoch_average_recall.reset_state()


# def load_raw_recording(file_path):
#     data = mne.io.read_raw_edf(file_path)
#     raw_data = data.get_data()
#     info = data.info
#     channels = data.ch_names
#
#     return raw_data


# def load_labels(labels_path, artifact_to_stages=False, just_artifact_labels=False):
#     df = pd.read_csv(labels_path, header=None)
#
#     labels_1 = pd.get_dummies(df[1])
#     labels_2 = pd.get_dummies(df[2])
#
#     # column names: {1, 2, 3, n, r, w}
#     # 1=wake artifact, 2=NREM artifact, 3=REM artifact
#
#     if artifact_to_stages:
#         labels_1.loc[labels_1["1"] == 1, 'w'] = 1
#         labels_2.loc[labels_2["1"] == 1, 'w'] = 1
#         labels_1.loc[labels_1["2"] == 1, 'n'] = 1
#         labels_2.loc[labels_2["2"] == 1, 'n'] = 1
#         labels_1.loc[labels_1["3"] == 1, 'r'] = 1
#         labels_2.loc[labels_2["3"] == 1, 'r'] = 1
#
#         labels_1 = labels_1.iloc[:, -3:]
#         labels_2 = labels_2.iloc[:, -3:]
#     elif just_artifact_labels:
#         labels_1.loc[(labels_1['1'] == 1) | (labels_1['2'] == 1) | (labels_1['3'] == 1), 'art'] = 1
#         labels_1.loc[(labels_1['1'] == 0) & (labels_1['2'] == 0) & (labels_1['3'] == 0), 'art'] = 0
#
#         labels_2.loc[(labels_2['1'] == 1) | (labels_2['2'] == 1) | (labels_2['3'] == 1), 'art'] = 1
#         labels_2.loc[(labels_2['1'] == 0) & (labels_2['2'] == 0) & (labels_2['3'] == 0), 'art'] = 0
#
#         labels_1 = labels_1['art']
#         labels_2 = labels_2['art']
#
#     return [labels_1, labels_2]


# def preprocess_EEG(signal,
#                    fs=128,
#                    stft_size=256,
#                    stft_stride=16,
#                    lowcut=0.5,
#                    highcut=24,
#                    visualize=False,
#                    labels=None,
#                    plot_artifacts=False):
#     if visualize:
#         # Select random epoch
#         rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
#         rdm_epoch_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]
#
#         if plot_artifacts:
#             labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
#         else:
#             labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}
#         rdm_epoch_labels = np.where(rdm_epoch_labels == 1)[1]
#         rdm_epoch_labels = [labels_dict[i] for i in rdm_epoch_labels]
#
#         rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
#         time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs
#
#         fig, ax = plt.subplots(6, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
#         fig.subplots_adjust(hspace=0.8)
#         cax = ax[0, 0]
#         cax.plot(time_axis, rdm_epoch_signal)
#         cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         cax.set_title('Raw 5 epochs window')
#         # cax.set_xlabel('Time (s)')
#         cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
#         cax.set_xlim((time_axis[0], time_axis[-1]))
#         epoch_labels_ax = cax.twiny()
#         epoch_labels_ax.set_xlim(cax.get_xlim())
#         epoch_labels_ax.set_xticks(np.linspace(time_axis[0]+ 2, time_axis[-1]-2, 5))
#         epoch_labels_ax.set_xticklabels(rdm_epoch_labels)
#         epoch_labels_ax.tick_params(length=0)
#         ax[0, 1].axis('off')
#
#     # STFT
#     f, t, Z = scipy.signal.stft(signal,
#                                 fs=128,
#                                 window='hamming',
#                                 nperseg=stft_size,
#                                 noverlap=stft_size - stft_stride
#                                 )
#
#     if visualize:
#         cax = ax[1, 0]
#
#         rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#         # time_axis = np.linspace((rdm_epoch_idx-2)*32, (rdm_epoch_idx+3)*32, 32*5)
#         time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('Spectrogram')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # Bandpass (crop)
#     Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]
#
#     if visualize:
#         cax = ax[2, 0]
#
#         rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('Bandpass')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # PSD
#     y = np.abs(Z) ** 2
#
#     if visualize:
#         cax = ax[3, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('PSD')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # Log-scale
#     y = 10 * np.log10(y)
#
#     if visualize:
#         cax = ax[4, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
#         cax.set_title('Log transformation')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
#
#     # Standardize
#     y_mean = np.mean(y, axis=1, keepdims=True)
#     y_std = np.std(y, axis=1, keepdims=True)
#
#     y = (y - y_mean) / y_std
#
#     if visualize:
#         cax = ax[5, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
#         cax.set_title('Standardization')
#         cax.invert_yaxis()
#         cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
#         plt.show()
#
#     return y


# def preprocess_EMG(signal,
#                    fs=128,
#                    stft_size=256,
#                    stft_stride=16,
#                    lowcut=0.5,
#                    highcut=30,
#                    visualize=False,
#                    labels=None,
#                    plot_artifacts=False):
#
#     if visualize:
#         # Select random epoch
#         rdm_epoch_idx = np.random.randint(2, len(signal) / 4 / fs - 2)
#         rdm_epoch_labels = labels.to_numpy()[rdm_epoch_idx - 2:rdm_epoch_idx + 3, :]
#
#         if plot_artifacts:
#             labels_dict = {0: 'W_art', 1: 'N_art', 2: 'R_art', 3: 'NREM', 4: 'REM', 5: 'WAKE'}
#         else:
#             labels_dict = {0: 'NREM', 1: 'REM', 2: 'WAKE'}
#         rdm_epoch_labels = np.where(rdm_epoch_labels == 1)[1]
#         rdm_epoch_labels = [labels_dict[i] for i in rdm_epoch_labels]
#
#         rdm_epoch_signal = signal[(rdm_epoch_idx - 2) * fs * 4: (rdm_epoch_idx + 3) * fs * 4]
#         time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, fs * 4 * 5) / fs
#
#         fig, ax = plt.subplots(7, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [60, 1]})
#         fig.subplots_adjust(hspace=0.8)
#         cax = ax[0, 0]
#         cax.plot(time_axis, rdm_epoch_signal)
#         cax.vlines(x=np.linspace(time_axis[0] + 4, time_axis[-1] - 4, 4),
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         cax.set_title('Raw 5 epochs window')
#         # cax.set_xlabel('Time (s)')
#         cax.set_xticks(np.linspace(time_axis[0], time_axis[-1], 6))
#         cax.set_xlim((time_axis[0], time_axis[-1]))
#         epoch_labels_ax = cax.twiny()
#         epoch_labels_ax.set_xlim(cax.get_xlim())
#         epoch_labels_ax.set_xticks(np.linspace(time_axis[0]+ 2, time_axis[-1]-2, 5))
#         epoch_labels_ax.set_xticklabels(rdm_epoch_labels)
#         epoch_labels_ax.tick_params(length=0)
#         ax[0, 1].axis('off')
#
#     # STFT
#     f, t, Z = scipy.signal.stft(signal,
#                                 fs=128,
#                                 window='hamming',
#                                 nperseg=stft_size,
#                                 noverlap=stft_size - stft_stride
#                                 )
#
#     if visualize:
#         cax = ax[1, 0]
#
#         rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#         # time_axis = np.linspace((rdm_epoch_idx-2)*32, (rdm_epoch_idx+3)*32, 32*5)
#         time_axis = np.linspace((rdm_epoch_idx - 2) * fs * 4, (rdm_epoch_idx + 3) * fs * 4, 18) / fs
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('Spectrogram')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels([str(f[-1]), str(f[-1] / 2), str(f[0])])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[1, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # Bandpass (crop)
#     Z = Z[np.where(f == lowcut)[0][0]: np.where(f == highcut)[0][0] + 1, :]
#
#     if visualize:
#         cax = ax[2, 0]
#
#         rdm_epoch_spect = Z[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('Bandpass')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[2, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # PSD
#     y = np.abs(Z) ** 2
#
#     if visualize:
#         cax = ax[3, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('PSD')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[3, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # Integration
#     y = np.sum(y, axis=0)
#
#     # Stack rows to have 2 dimensions
#     y = np.expand_dims(y, axis=0)
#     # y = np.repeat(y, eeg_dimensions[0], axis=0)
#     y = np.repeat(y, 48, axis=0)
#
#     if visualize:
#         cax = ax[4, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(np.abs(rdm_epoch_spect), cmap='jet', aspect='auto')
#         cax.set_title('Integration')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[4, 1], ticks=[np.min(np.abs(rdm_epoch_spect)), np.max(np.abs(rdm_epoch_spect))])
#
#     # Log-scale
#     y = 10*np.log10(y)
#
#     if visualize:
#         cax = ax[5, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
#         cax.set_title('Log transformation')
#         cax.invert_yaxis()
#         # cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[5, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
#
#     # Standardize
#     y_mean = np.mean(y, axis=1, keepdims=True)
#     y_std = np.std(y, axis=1, keepdims=True)
#
#     y = (y - y_mean) / y_std
#
#     if visualize:
#         cax = ax[6, 0]
#
#         rdm_epoch_spect = y[:, (rdm_epoch_idx-2) * 32 : (rdm_epoch_idx+3) * 32]
#
#         img = cax.imshow(rdm_epoch_spect, cmap='jet', aspect='auto')
#         cax.set_title('Standardization')
#         cax.invert_yaxis()
#         cax.set_xlabel('Time (s)')
#         cax.set_ylabel('Frequency (Hz.)')
#         cax.set_xticks(np.linspace(cax.get_xlim()[0], cax.get_xlim()[1], 6))
#         cax.set_xticklabels(np.linspace(time_axis[0], time_axis[-1], 6, dtype='int'))
#         cax.set_yticks([cax.get_ylim()[1], abs(cax.get_ylim()[1] - cax.get_ylim()[0]) / 2, cax.get_ylim()[0]])
#         cax.set_yticklabels(['24', '12', '0'])
#         cax.vlines(x=cax.get_xticks()[1:-1],
#                    ymin=cax.get_ylim()[0], ymax=cax.get_ylim()[1], color='k')
#         fig.colorbar(img, cax=ax[6, 1], ticks=[np.min(rdm_epoch_spect), np.max(rdm_epoch_spect)])
#         plt.show()
#
#     return y


# def windowing(signal, window_size=32*5, window_stride=32, fs=128):
#     n_windows = 21600 - 4
#
#     windowed_signal = np.zeros((n_windows, 3, 48, window_size))
#
#     # signal = signal[window_size//2 : -window_size//2],
#
#     for i in range(n_windows):
#         windowed_signal[i, :, :, :] = signal[:, :, (i*window_stride) : (i*window_stride) + window_size]
#
#     return windowed_signal


# def load_recording_to_dataset_2(signal_path, labels_path, validation_split=None): # stft_size, stft_stride, fs, epoch_length,
#     y_1, y_2 = load_labels(labels_path, artifact_to_stages=True)
#     # filter_epochs(y_1, y_2)
#     raw_data = load_raw_recording(signal_path)
#     eeg_1 = preprocess_EEG(raw_data[0, :], labels=y_1)  # , visualize=True)
#     eeg_2 = preprocess_EEG(raw_data[1, :], labels=y_1)  # , visualize=True)
#     emg = preprocess_EMG(raw_data[2, :], labels=y_1)  # , visualize=True)
#     x = np.stack((eeg_1, eeg_2, emg))
#
#     x = windowing(x, window_size=32 * 5, window_stride=32)
#     x = np.transpose(x, (0, 3, 2, 1))
#
#     if validation_split is not None:
#         rdm_indexes = np.arange(x.shape[0])
#         np.random.shuffle(rdm_indexes)
#         train_indexes = rdm_indexes[:int(len(rdm_indexes)*(1-validation_split))]
#         val_indexes = rdm_indexes[int(len(rdm_indexes)*(1-validation_split)):]
#
#         x_train = x[train_indexes]
#         labels_train = y_1.to_numpy()[2:-2][train_indexes]
#         x_val = x[val_indexes]
#         labels_val = y_1.to_numpy()[2:-2][val_indexes]
#
#         input_dataset_train = tf.data.Dataset.from_tensor_slices(x_train)
#         labels_dataset_train = tf.data.Dataset.from_tensor_slices(labels_train)
#         input_dataset_val = tf.data.Dataset.from_tensor_slices(x_val)
#         labels_dataset_val = tf.data.Dataset.from_tensor_slices(labels_val)
#         # labels_dataset = labels_dataset.batch(32)
#
#         train_dataset = tf.data.Dataset.zip((input_dataset_train, labels_dataset_train))
#         val_dataset = tf.data.Dataset.zip((input_dataset_val, labels_dataset_val))
#
#         return train_dataset, val_dataset
#     else:
#         input_dataset = tf.data.Dataset.from_tensor_slices(x)
#         labels_dataset = tf.data.Dataset.from_tensor_slices(y_1[2:-2])
#         dataset = tf.data.Dataset.zip((input_dataset, labels_dataset))
#
#         return dataset
