from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from spindle_data.spindle_data_loading import load_to_dataset
from metrics import *
from tools import *
plt.ion()

saved_model_folder = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE_pycharm\results\3 - new round of results after meeting'
model_name = 'B_1'


# -------------------------------------------------------------------------------------------------------------------------

signal_paths = [r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A3.edf",
                r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A4.edf"]
labels_paths = [r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A3.csv',
                r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A4.csv']


test_dataset = load_to_dataset(signal_paths=signal_paths,
                               labels_paths=labels_paths,
                               scorer=0,
                               just_artifact_labels=True,
                               artifact_to_stages=False,
                               balance_artifacts=False,
                               validation_split=0)

batch_size = 100
test_dataset = test_dataset.batch(batch_size)

# -------------------------------------------------------------------------------------------------------------------------

# test_sequence = SequenceDataset(r'C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\preprocessed\d2\testing\Cohort A',
#                                 batch_size=100,
#                                 binary=False)

# -------------------------------------------------------------------------------------------------------------------------

spindle_model = tf.keras.Sequential([
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

spindle_model.load_weights(os.path.join(saved_model_folder, model_name, model_name + "_5epochs" + ".h5"))

spindle_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss=BinaryWeightedCrossEntropy,
                      # BinaryWeightedCrossEntropy tf.keras.losses.BinaryCrossentropy()
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               BinaryBalancedAccuracy(),
                               BinaryF1Score()])


for idx, batch in enumerate(test_dataset):
    if idx == 0:
        y_pred = spindle_model(batch[0])
        y_true = batch[1]
    else:
        y_pred = tf.concat([y_pred, spindle_model(batch[0])], axis=0)
        y_true = tf.concat([y_true, batch[1]], axis=0)

# for idx in range(len(test_sequence)):
#     batch = test_sequence[idx]
#     if idx == 0:
#         y_pred = spindle_model(batch[0])
#         y_true = batch[1]
#     else:
#         y_pred = tf.concat([y_pred, spindle_model(batch[0])], axis=0)
#         y_true = tf.concat([y_true, batch[1]], axis=0)


# ------------------------------------------------ THR=0.5 -------------------------------------------------------------

thr = 0.5

save_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE_pycharm\results\3 - new round of results after meeting\B_1\Evaluation\Against intersection'
compute_and_save_metrics_cnn2(y_true, y_pred, thr, save_path, model_name)

# ------------------------------------------------ THR=optimal----------------------------------------------------------





############################################ REMOVED FUNCTIONS ####################################################

# class BinaryF1Score(tf.keras.metrics.Metric):
#     '''
#     Implementation as in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
#     https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
#     https://en.wikipedia.org/wiki/Confusion_matrix
#
#     Is the average of the f1-score of each batch.
#
#     I tested it and gives the same result as sklearn.metrics.balanced_accuracy_score(y_true, y_pred).
#     Code to test it:
#         y_true1 = np.array([0, 0, 0, 0, 1, 1, 1])
#         y_pred1 = np.array([0, 0, 1, 0, 1, 0, 1])
#         y_true2 = np.array([0, 1, 0, 0, 0, 0, 1])
#         y_pred2 = np.array([0, 0, 1, 0, 1, 0, 1])
#         av_bacc = (sklearn.metrics.f1_score(y_true1, y_pred1) + sklearn.metrics.f1_score(y_true2, y_pred2)) / 2
#         print(av_bacc)
#         y_true1 = tf.convert_to_tensor(y_true1, dtype=tf.float64)
#         y_pred1 = tf.convert_to_tensor(y_pred1, dtype=tf.float64)
#         y_true2 = tf.convert_to_tensor(y_true2, dtype=tf.float64)
#         y_pred2 = tf.convert_to_tensor(y_pred2, dtype=tf.float64)
#         BBA = BinaryF1Score()
#         BBA.update_state(y_true1, y_pred1)
#         BBA.update_state(y_true2, y_pred2)
#         print(BBA.result())
#     '''
#
#     def __init__(self, name='F1_score', **kwargs):
#         super(BinaryF1Score, self).__init__(name=name, **kwargs)
#         self.TP = tf.keras.metrics.TruePositives()
#         self.FP = tf.keras.metrics.FalsePositives()
#         self.FN = tf.keras.metrics.FalseNegatives()
#         self.epoch_average_F1 = tf.keras.metrics.Mean()
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#
#         # tf.print(y_true.dtype)
#         # tf.print((y_pred>0.5).dtype)
#
#         self.TP.reset_state()
#         self.FP.reset_state()
#         self.FN.reset_state()
#
#         self.TP.update_state(y_true>0.5, y_pred>0.5)
#         self.FP.update_state(y_true>0.5, y_pred>0.5)
#         self.FN.update_state(y_true>0.5, y_pred>0.5)
#
#         f1_score = 2*self.TP.result() / (2*self.TP.result() + self.FP.result() + self.FN.result())
#
#         self.epoch_average_F1.update_state(f1_score)
#
#     def result(self):
#         return self.epoch_average_F1.result()
#
#     def reset_state(self):
#         self.epoch_average_F1.reset_state()


# class BinaryBalancedAccuracy(tf.keras.metrics.Metric):
#     '''
#     Gives the average of the balanced accuracy over each batch.
#
#     Same implementation as in https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/metrics/_classification.py#L1933
#
#     I tested it and gives the same result as sklearn.metrics.balanced_accuracy_score(y_true, y_pred).
#     Code to test it:
#         y_true1 = np.array([0, 0, 0, 0, 1, 1, 1])
#         y_pred1 = np.array([0, 0, 1, 0, 1, 0, 1])
#         y_true2 = np.array([0, 1, 0, 0, 0, 0, 1])
#         y_pred2 = np.array([0, 0, 1, 0, 1, 0, 1])
#         av_bacc = (sklearn.metrics.balanced_accuracy_score(y_true1, y_pred1) + sklearn.metrics.balanced_accuracy_score(y_true2, y_pred2))/2
#         print(av_bacc)
#         y_true1 = tf.convert_to_tensor(y_true1)
#         y_pred1 = tf.convert_to_tensor(y_pred1)
#         y_true2 = tf.convert_to_tensor(y_true2)
#         y_pred2 = tf.convert_to_tensor(y_pred2)
#         BBA = BinaryBalancedAccuracy()
#         BBA.update_state(y_true1, y_pred1)
#         BBA.update_state(y_true2, y_pred2)
#         print(BBA.result())
#     '''
#
#     def __init__(self, name='binary_balanced_accuracy', **kwargs):
#         super(BinaryBalancedAccuracy, self).__init__(name=name, **kwargs)
#         self.TP = tf.keras.metrics.TruePositives()
#         self.FN = tf.keras.metrics.FalseNegatives()
#         self.FP = tf.keras.metrics.FalsePositives()
#         self.TN = tf.keras.metrics.TrueNegatives()
#         self.epoch_average_accuracy = tf.keras.metrics.Mean()
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#
#         self.TP.reset_state()
#         self.FN.reset_state()
#         self.FP.reset_state()
#         self.TN.reset_state()
#
#         self.TP.update_state(y_true, y_pred)
#         self.FN.update_state(y_true, y_pred)
#         self.FP.update_state(y_true, y_pred)
#         self.TN.update_state(y_true, y_pred)
#
#         sensitivity = self.TP.result() / (self.TP.result() + self.FN.result())
#         specificity = self.TN.result() / (self.TN.result() + self.FP.result())
#
#         balanced_accuracy = (sensitivity + specificity)/2
#
#         self.epoch_average_accuracy.update_state(balanced_accuracy)
#
#     def result(self):
#         return self.epoch_average_accuracy.result()
#
#     def reset_state(self):
#         self.epoch_average_accuracy.reset_state()


# class BinaryWeightedCrossEntropy(tf.keras.losses.Loss):
#     '''
#     Same implementation as in https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
#
#     Weights are calculated as w_i = n_samples / (n_classes * n_elements_class_i)
#     '''
#
#     def __init__(self, name="binary_weighted_cross_entropy"):
#         super().__init__(name=name)
#
#     def call(self, y_true, y_pred):
#
#         ce = tf.keras.metrics.binary_crossentropy(y_true, y_pred)
#         # tf.print('ce: ', ce)
#
#         for i in range(2):
#             num_class_i = tf.shape(tf.where(y_true == i))[0]
#             # tf.print('num_class_i: ', num_class_i)
#
#             if num_class_i > 0:
#                 w = tf.shape(y_true)[0] / (2 * num_class_i)
#                 w = tf.cast(w, dtype=tf.float32)
#                 # tf.print('weight: ', w)
#
#                 indexes = y_true == i
#
#                 ce = tf.tensor_scatter_nd_update(ce, tf.expand_dims(tf.where(y_true == i)[:, 0], axis=1), w * ce[indexes[:, 0]])
#
#                 # tf.print('ce updated')
#
#         # tf.print('ce_updated: ', ce)
#         return ce
