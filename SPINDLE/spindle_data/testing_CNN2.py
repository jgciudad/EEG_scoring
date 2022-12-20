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

save_path = r'/results/3 - new round of results after meeting/B_1/Evaluation/Against intersection'
compute_and_save_metrics_cnn2(y_true, y_pred, thr, save_path, model_name)

# ------------------------------------------------ THR=optimal----------------------------------------------------------





