from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Dense, Flatten, Dropout
from spindle_data.spindle_data_loading import load_to_dataset
from metrics import *
from tools import *
from hmm import *

saved_model_folder = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE_pycharm\results\3 - new round of results after meeting'
# save_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE\results\3 - new round of results after meeting\A_1\Evaluation\Against scorers intersection\Excluding artifacts\Before HMM'
model_name = 'A_1'

plt.ion()

# -------------------------------------------------------------------------------------------------------------------------

test_dataset_1 = load_to_dataset(signal_paths=[r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A3.edf"],
                               labels_paths=[r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A3.csv'],
                               scorer=1,
                               just_artifact_labels=False,
                               artifact_to_stages=False,
                               balance_artifacts=False,
                               validation_split=0)

test_dataset_2 = load_to_dataset(signal_paths=[r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A4.edf"],
                               labels_paths=[r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A4.csv'],
                               scorer=1,
                               just_artifact_labels=False,
                               artifact_to_stages=False,
                               balance_artifacts=False,
                               validation_split=0)

test_dataset_3 = load_to_dataset(signal_paths=[r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A3.edf"],
                               labels_paths=[r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A3.csv'],
                               scorer=1,
                               just_artifact_labels=False,
                               artifact_to_stages=False,
                               balance_artifacts=False,
                               validation_split=0)

test_dataset_4 = load_to_dataset(signal_paths=[r"C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/recordings/A4.edf"],
                               labels_paths=[r'C:/Users/javig/Desktop/SPINDLE dataset/SPINDLE dataset/data (original)/CohortA/scorings/A4.csv'],
                               scorer=1,
                               just_artifact_labels=False,
                               artifact_to_stages=False,
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
    Dense(units=3, activation='softmax', kernel_initializer='glorot_uniform')
])


spindle_model.load_weights(os.path.join(saved_model_folder, model_name, model_name + "_5epochs" + ".h5"))

spindle_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5 * 1e-5,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss=MulticlassWeightedCrossEntropy_2(n_classes=3),
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               BinaryBalancedAccuracy(),
                               BinaryF1Score()])


for idx, batch in enumerate(test_dataset_3):
    if idx == 0:
        y_pred_3 = spindle_model(batch[0])
        y_true_3 = batch[1]
    else:
        y_pred_3 = tf.concat([y_pred_3, spindle_model(batch[0])], axis=0)
        y_true_3 = tf.concat([y_true_3, batch[1]], axis=0)

for idx, batch in enumerate(test_dataset_4):
    if idx == 0:
        y_pred_4 = spindle_model(batch[0])
        y_true_4 = batch[1]
    else:
        y_pred_4 = tf.concat([y_pred_4, spindle_model(batch[0])], axis=0)
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
y_pred_cnn_3 = np.argmax(y_pred_3, axis=1)
y_true_4 = np.argmax(y_true_4, axis=1)
y_pred_cnn_4 = np.argmax(y_pred_4, axis=1)


# ------------------------------------------------ BEFORE HMM ----------------------------------------------------------
#
save_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE\results\3 - new round of results after meeting\A_1\Evaluation\Spindle data\Against scorer 2\Before HMM'
compute_and_save_metrics_cnn1(np.hstack((y_true_3, y_true_4)), np.hstack((y_pred_cnn_3, y_pred_cnn_4)), save_path, model_name)

# ------------------------------------------------ AFTER HMM -----------------------------------------------------------

transition_matrix = get_transition_matrix([r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\scorings\A1.csv",
                                           r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\scorings\A2.csv"],
                                           cancel_forbidden_transitions=True)

# initial_probs = np.array([1/3, 1/3, 1/3])
initial_probs = get_priors([r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\scorings\A1.csv",
                          r"C:\Users\javig\Desktop\SPINDLE dataset\SPINDLE dataset\data (original)\CohortA\scorings\A2.csv"])

y_pred_hmm_3, _, _ = viterbi(y_pred_3, transition_matrix, initial_probs)
y_pred_hmm_4, _, _ = viterbi(y_pred_4, transition_matrix, initial_probs)

n_corrected_epochs_3, correction_matrix_3, cnn_transitions_3, hmm_transitions_3 = evaluate_hmm_effect(y_true_3, y_pred_cnn_3, y_pred_hmm_3)
n_corrected_epochs_4, correction_matrix_4, cnn_transitions_4, hmm_transitions_4 = evaluate_hmm_effect(y_true_4, y_pred_cnn_4, y_pred_hmm_4)

save_path = r'C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\SPINDLE\results\3 - new round of results after meeting\A_1\Evaluation\Spindle data\Against scorer 2\After HMM'
plot_and_save_hmm_effect(n_corrected_epochs_3 + n_corrected_epochs_4,
                         correction_matrix_3 + correction_matrix_4,
                         cnn_transitions_3 + cnn_transitions_4,
                         hmm_transitions_3 + hmm_transitions_4,
                         save_path, model_name)

# CONCATENATE
compute_and_save_metrics_cnn1(np.hstack((y_true_3, y_true_4)), np.hstack((y_pred_hmm_3, y_pred_hmm_4)), save_path, model_name)







