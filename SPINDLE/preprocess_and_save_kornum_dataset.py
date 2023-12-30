from kornum_data.kornum_data_loading import load_recording
import os
import string
import random
import pandas as pd
import numpy as np

dataset_folder = '/Users/tlj258/Documents/Laura-EEGdata_cleaned'
destination_folder = '/Users/tlj258/spindle_data'


training_validation_labels = [
    '2DTUSERVER-Alexandra/tsv/M23-b1.tsv',
    '2DTUSERVER-Alexandra/tsv/M23-b2.tsv',
    '2DTUSERVER-Alexandra/tsv/M23-b3.tsv',
    '2DTUSERVER-Alexandra/tsv/M29-b1.tsv',
    '2DTUSERVER-Alexandra/tsv/M29-b2.tsv',
    '2DTUSERVER-Alexandra/tsv/M29-b3.tsv',
    '2DTUSERVER-Alexandra/tsv/M48-b1.tsv',
    '2DTUSERVER-Alexandra/tsv/M48-b2.tsv',
    '2DTUSERVER-Alexandra/tsv/M48-b3.tsv',
    '2DTUSERVER-Alexandra/tsv/M52-b1.tsv',
    '2DTUSERVER-Alexandra/tsv/M52-b3.tsv',
    '2DTUSERVER-Alexandra/tsv/M58-b1.tsv',
    '2DTUSERVER-Alexandra/tsv/M58-b3.tsv',
    '2DTUSERVER-CH/tsv/m1-b1.tsv',
    '2DTUSERVER-CH/tsv/m11-b1.tsv',
    '2DTUSERVER-CH/tsv/m12-b1.tsv',
    '2DTUSERVER-CH/tsv/m14-b1.tsv',
    '2DTUSERVER-CH/tsv/m14-r3.tsv',
    '2DTUSERVER-CH/tsv/m15-r3.tsv',
    '2DTUSERVER-CH/tsv/m2-b1.tsv',
    '2DTUSERVER-CH/tsv/m3-b1.tsv',
    '2DTUSERVER-CH/tsv/m4-b1.tsv',
    '2DTUSERVER-CH/tsv/m5-b1.tsv',
    '2DTUSERVER-CH/tsv/m6-r3.tsv',
    '2DTUSERVER-CH/tsv/m7-b1.tsv',
    '2DTUSERVER-CH/tsv/m8-b1.tsv',
    '2DTUSERVER-CH/tsv/m8-r3.tsv',
    '2DTUSERVER-LOUISE/tsv/M16-b2.tsv',
    '2DTUSERVER-LOUISE/tsv/M16-b3.tsv',
    '2DTUSERVER-LOUISE/tsv/M18-b3.tsv',
    '2DTUSERVER-LOUISE/tsv/M20-b3.tsv',
    '2DTUSERVER-LOUISE/tsv/M309-b1.tsv',
    '2DTUSERVER-Maria/tsv/m121-b1.tsv',
    '2DTUSERVER-Maria/tsv/m121-b2.tsv',
    '2DTUSERVER-Maria/tsv/m61-b1.tsv',
    '2DTUSERVER-Maria/tsv/m63-b1.tsv',
    '2DTUSERVER-Maria/tsv/m63-b2.tsv',
    '2DTUSERVER-Maria/tsv/m86-b1.tsv',
    '2DTUSERVER-Maria/tsv/m88-b1.tsv',
    '2DTUSERVER-Maria/tsv/m88-b2.tsv',
    '2DTUSERVER-Maria/tsv/m96-b1.tsv',
    '2DTUSERVER-Maria/tsv/m96-b2.tsv',
    '2DTUSERVER-Marieke/tsv/m2-b1.tsv',
    '2DTUSERVER-Marieke/tsv/m21-b1.tsv'
    ]

training_validation_signals = [
    '2DTUSERVER-Alexandra/edf/M23-b1.edf',
    '2DTUSERVER-Alexandra/edf/M23-b2.edf',
    '2DTUSERVER-Alexandra/edf/M23-b3.edf',
    '2DTUSERVER-Alexandra/edf/M29-b1.edf',
    '2DTUSERVER-Alexandra/edf/M29-b2.edf',
    '2DTUSERVER-Alexandra/edf/M29-b3.edf',
    '2DTUSERVER-Alexandra/edf/M48-b1.edf',
    '2DTUSERVER-Alexandra/edf/M48-b2.edf',
    '2DTUSERVER-Alexandra/edf/M48-b3.edf',
    '2DTUSERVER-Alexandra/edf/M52-b1.edf',
    '2DTUSERVER-Alexandra/edf/M52-b3.edf',
    '2DTUSERVER-Alexandra/edf/M58-b1.edf',
    '2DTUSERVER-Alexandra/edf/M58-b3.edf',
    '2DTUSERVER-CH/edf/m1-b1.edf',
    '2DTUSERVER-CH/edf/m11-b1.edf',
    '2DTUSERVER-CH/edf/m12-b1.edf',
    '2DTUSERVER-CH/edf/m14-b1.edf',
    '2DTUSERVER-CH/edf/m14-r3.edf',
    '2DTUSERVER-CH/edf/m15-r3.edf',
    '2DTUSERVER-CH/edf/m2-b1.edf',
    '2DTUSERVER-CH/edf/m3-b1.edf',
    '2DTUSERVER-CH/edf/m4-b1.edf',
    '2DTUSERVER-CH/edf/m5-b1.edf',
    '2DTUSERVER-CH/edf/m6-r3.edf',
    '2DTUSERVER-CH/edf/m7-b1.edf',
    '2DTUSERVER-CH/edf/m8-b1.edf',
    '2DTUSERVER-CH/edf/m8-r3.edf',
    '2DTUSERVER-LOUISE/edf/M16-b2.edf',
    '2DTUSERVER-LOUISE/edf/M16-b3.edf',
    '2DTUSERVER-LOUISE/edf/M18-b3.edf',
    '2DTUSERVER-LOUISE/edf/M20-b3.edf',
    '2DTUSERVER-LOUISE/edf/M309-b1.edf',
    '2DTUSERVER-Maria/edf/m121-b1.edf',
    '2DTUSERVER-Maria/edf/m121-b2.edf',
    '2DTUSERVER-Maria/edf/m61-b1.edf',
    '2DTUSERVER-Maria/edf/m63-b1.edf',
    '2DTUSERVER-Maria/edf/m63-b2.edf',
    '2DTUSERVER-Maria/edf/m86-b1.edf',
    '2DTUSERVER-Maria/edf/m88-b1.edf',
    '2DTUSERVER-Maria/edf/m88-b2.edf',
    '2DTUSERVER-Maria/edf/m96-b1.edf',
    '2DTUSERVER-Maria/edf/m96-b2.edf',
    '2DTUSERVER-Marieke/edf/m2-b1.edf',
    '2DTUSERVER-Marieke/edf/m21-b1.edf'
    ]

test_signals = [
    '2DTUSERVER-Alexandra/EDF/M52-b2.edf',
    '2DTUSERVER-Alexandra/EDF/M58-b2.edf',
    '2DTUSERVER-CH/edf/m13-b1.edf',
    '2DTUSERVER-CH/EDF/m15-b1.edf',
    '2DTUSERVER-CH/edf/m6-b1.edf',
    '2DTUSERVER-LOUISE/EDF/M18-b2.edf',
    '2DTUSERVER-LOUISE/EDF/M313-b1.edf',
    '2DTUSERVER-Maria/EDF/m61-b2.edf',
    '2DTUSERVER-Maria/edf/m86-b2.edf',
    '2DTUSERVER-Maria/EDF/m94-b1.edf',
    '2DTUSERVER-Maria/edf/m94-b2.edf'
    ]

test_labels = [
    '2DTUSERVER-Alexandra/tsv/M52-b2.tsv',
    '2DTUSERVER-Alexandra/tsv/M58-b2.tsv',
    '2DTUSERVER-CH/tsv/m13-b1.tsv',
    '2DTUSERVER-CH/tsv/m15-b1.tsv',
    '2DTUSERVER-CH/tsv/m6-b1.tsv',
    '2DTUSERVER-LOUISE/tsv/M18-b2.tsv',
    '2DTUSERVER-LOUISE/tsv/M313-b1.tsv',
    '2DTUSERVER-Maria/tsv/m61-b2.tsv',
    '2DTUSERVER-Maria/tsv/m86-b2.tsv',
    '2DTUSERVER-Maria/tsv/m94-b1.tsv',
    '2DTUSERVER-Maria/tsv/m94-b2.tsv'
    ]


def save_to_numpy(data, labels, path, df_all, set):
    filenames = []

    for idx, r in enumerate(data):
        characters = string.ascii_lowercase + string.digits
        filename = ''.join(random.choice(characters) for i in range(16))
        while os.path.exists(os.path.join(path, filename) + '.npy'):
            filename = ''.join(random.choice(characters) for i in range(16))

        np.save(os.path.join(path, filename), r)
        filenames.append(filename)

    df = pd.DataFrame(labels, columns=['NREM', 'REM', 'WAKE', 'Art'])
    filenames = [f + '.npy' for f in filenames]
    df.insert(0, 'File', filenames)

    if set == 'train':
        df['train'] = 1
        df['validation'] = 0
        df['test'] = 0
    elif set == 'validation':
        df['train'] = 0
        df['validation'] = 1
        df['test'] = 0
    elif set == 'test':
        df['train'] = 0
        df['validation'] = 0
        df['test'] = 1

    if df_all is not None:
        df_all = pd.concat([df_all, df])
    else:
        df_all = df

    return df_all


number_of_files = len(training_validation_signals) + len(test_signals)
file_counter = 1

for i in range(len(test_signals)):
    print('Processing file ', file_counter)
    print('Remaning files: ', number_of_files-file_counter)

    x, y = load_recording(dataset_folder + os.sep + test_signals[i],
                          dataset_folder + os.sep + test_labels[i],
                          resample_rate=128,
                          just_artifact_labels=False,
                          just_stage_labels=False,
                          validation_split=0)

    if i == 0:
        df_all = None

    df_all = save_to_numpy(x, y, destination_folder, df_all, 'test')

    file_counter += 1

for i in range(len(training_validation_signals)):
    print('Processing file ', file_counter)
    print('Remaning files: ', number_of_files-file_counter)

    x_train, x_val, labels_train, labels_val = load_recording(dataset_folder + os.sep + training_validation_signals[i],
                                                              dataset_folder + os.sep + training_validation_labels[i],
                                                              resample_rate=128,
                                                              just_artifact_labels=False,
                                                              just_stage_labels=False,
                                                              validation_split=0.15)

    if i==0:
        df_all = None

    df_all = save_to_numpy(x_train, labels_train, destination_folder, df_all, 'train')
    df_all = save_to_numpy(x_val, labels_val, destination_folder, df_all, 'validation')

    file_counter += 1

df_all.to_csv(os.path.dirname(destination_folder) + '/labels_all.csv', index=False)


