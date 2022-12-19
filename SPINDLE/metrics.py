import tensorflow as tf




def multiclass_f1_score(y_true, y_pred, average='macro'):
    '''
    Implementation as in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html, with average='macro' or 'weighted'
    https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    https://en.wikipedia.org/wiki/Confusion_matrix

    Is the average of the f1-score of each class. If weighted, it is the weighted average, where the weight is the
    prevalence of each class (smaller classes are penalized)

    I tested it and gives the same result as sklearn.metrics.f1_score(y_true, y_pred, average='weighted').
    Code to test it:
        y_true = np.array(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
        y_pred = np.array(
            [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sklearn.metrics.f1_score(y_true, y_pred, average='weighted')
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        multiclass_weighted_f1_score(y_true, y_pred)
    '''


    class_weights = tf.cast(tf.math.reduce_sum(y_true, axis=0), dtype=tf.int32) / tf.shape(y_true)[0]

    class_weights = tf.cast(class_weights, dtype=tf.float32)

    f1_score = tf.convert_to_tensor(0, dtype=tf.float32)
    for i in range(tf.shape(class_weights)[0]):

        w = class_weights[i]
        if w != 0:

            TPi, FPi, TNi, FNi = get_simple_metrics(y_true[y_true[:, i] == 1], y_pred[y_true[:, i] == 1])

            f1_i = 2*TPi / (2*TPi + FPi + FNi)

            if average == 'macro':
                f1_score = f1_score + f1_i
            elif average == 'weighted':
                f1_score = f1_score + w*f1_i

    if average=='macro':
        return f1_score/tf.math.count_nonzero(class_weights, dtype=tf.float32)
    elif average=='weighted':
        return f1_score


class MulticlassF1Score(tf.keras.metrics.Metric):
    '''
    Implementation as in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html, with average='macro'
    https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    https://en.wikipedia.org/wiki/Confusion_matrix

    Is the average of the f1-score of each batch, where the f1-score of a batch is the average of the f1-score across each
    class in that batch.

    I tested it and gives the same result as sklearn.metrics.balanced_accuracy_score(y_true, y_pred).
    Code to test it:
        f1_batch_1 = sklearn.metrics.f1_score(y_true1, y_pred1, average='macro')
        f1_batch_2 = sklearn.metrics.f1_score(y_true2, y_pred2, average='macro')
        av_f1 = (f1_batch_1 + f1_batch_2)/2
        y_true1 = np.array(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
        y_pred1 = np.array(
            [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_true2 = np.array(
            [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
        y_pred2 = np.array(
            [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_true1 = tf.convert_to_tensor(y_true1)
        y_pred1 = tf.convert_to_tensor(y_pred1)
        y_true2 = tf.convert_to_tensor(y_true2)
        y_pred2 = tf.convert_to_tensor(y_pred2)
        MBA = MulticlassF1Score(n_classes=3)
        MBA.update_state(y_true1, y_pred1)
        print(MBA.result())
        MBA.update_state(y_true2, y_pred2)
        print(MBA.result())
    '''

    def __init__(self, n_classes, name='multiclass_F1_score', **kwargs):
        super(MulticlassF1Score, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.TP = tf.keras.metrics.TruePositives()
        self.FP = tf.keras.metrics.FalsePositives()
        self.FN = tf.keras.metrics.FalseNegatives()
        self.epoch_average_F1 = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=1), self.n_classes)

        batch_F1_sum = tf.convert_to_tensor(0, dtype=tf.float32)

        for i in range(self.n_classes):
            if tf.shape(tf.where(y_true[:, i] == 1))[0] > 0:
                self.TP.reset_state()
                self.FP.reset_state()
                self.FN.reset_state()

                self.TP.update_state(y_true[:, i] == 1, y_pred[:, i] == 1)
                self.FP.update_state(y_true[:, i] == 1, y_pred[:, i] == 1)
                self.FN.update_state(y_true[:, i] == 1, y_pred[:, i] == 1)

                class_f1 = 2*self.TP.result() / (2*self.TP.result() + self.FP.result() + self.FN.result())

                batch_F1_sum = batch_F1_sum + class_f1

        batch_average_F1 = batch_F1_sum / tf.math.count_nonzero(tf.reduce_sum(y_true, axis=0), dtype=tf.float32)

        self.epoch_average_F1.update_state(batch_average_F1)

    def result(self):
        return self.epoch_average_F1.result()

    def reset_state(self):
        self.epoch_average_F1.reset_state()


def multiclass_balanced_accuracy(y_true, y_pred, n_classes=3):
    '''
    Is the average of the recall of each class.

    Same implementation as in https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/metrics/_classification.py#L1933

    I tested it and gives the same result as sklearn.metrics.balanced_accuracy_score(y_true, y_pred).
    Code to test it:
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 1, 0, 1, 2, 0])
        sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
        y_true = np.array(
                [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
        y_pred = np.array(
                [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        multiclass_balanced_accuracy(y_true, y_pred, 3)
    '''

    recalls = []
    for i in range(n_classes):
        if tf.shape(tf.where(y_true[:, i] == 1))[0] > 0:
            TPi, FPi, TNi, FNi = get_simple_metrics(y_true[y_true[:, i] == 1], y_pred[y_true[:, i] == 1])

            class_recall = TPi / (TPi+FNi)

            # tf.print('class_recall: ', class_recall)

            recalls.append(class_recall)

    recalls = tf.convert_to_tensor(recalls)

    average_recall = tf.math.reduce_mean(recalls)
    # tf.print('average_recall: ', tf.math.reduce_mean(recalls))

    return average_recall


class MulticlassBalancedAccuracy(tf.keras.metrics.Metric):
    '''
    Gives the average of the recall over each batch, where the recall of each batch is the average of the recall of each
    class in that batch.

    Same implementation as in https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/metrics/_classification.py#L1933

    I tested it and gives the same result as sklearn.metrics.balanced_accuracy_score(y_true, y_pred).
    Code to test it:
        y_true1 = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
        y_pred1 = np.array([0, 0, 1, 0, 1, 0, 1, 2, 0])
        y_true2 = np.array([0, 2, 0, 0, 0, 0, 1, 2, 2])
        y_pred2 = np.array([0, 0, 1, 0, 1, 0, 1, 2, 0])
        av_bacc = (sklearn.metrics.balanced_accuracy_score(y_true1, y_pred1) + sklearn.metrics.balanced_accuracy_score(y_true2, y_pred2))/2
        print(av_bacc)
        y_true1 = np.array(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
        y_pred1 = np.array(
            [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_true2 = np.array(
            [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
        y_pred2 = np.array(
            [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_true1 = tf.convert_to_tensor(y_true1)
        y_pred1 = tf.convert_to_tensor(y_pred1)
        y_true2 = tf.convert_to_tensor(y_true2)
        y_pred2 = tf.convert_to_tensor(y_pred2)
        MBA = MulticlassBalancedAccuracy(n_classes=3)
        MBA.update_state(y_true1,y_pred1)
        MBA.update_state(y_true2,y_pred2)
        print(MBA.result())
    '''

    def __init__(self, n_classes, name='multiclass_balanced_accuracy', **kwargs):
        super(MulticlassBalancedAccuracy, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.TP = tf.keras.metrics.TruePositives()
        self.FN = tf.keras.metrics.FalseNegatives()
        self.epoch_average_recall = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):

        batch_recalls_sum = tf.convert_to_tensor(0, dtype=tf.float32)

        for i in range(self.n_classes):
            if tf.shape(tf.where(y_true[:, i] == 1))[0] > 0:
                self.TP.reset_state()
                self.FN.reset_state()

                self.TP.update_state(y_true[y_true[:, i] == 1], y_pred[y_true[:, i] == 1])
                self.FN.update_state(y_true[y_true[:, i] == 1], y_pred[y_true[:, i] == 1])

                class_recall = self.TP.result() / (self.TP.result() + self.FN.result())

                batch_recalls_sum = batch_recalls_sum + class_recall

                # tf.print('class_recall: ', class_recall)

        batch_average_recall = batch_recalls_sum / tf.math.count_nonzero(tf.reduce_sum(y_true, axis=0), dtype=tf.float32)

        self.epoch_average_recall.update_state(batch_average_recall)

    def result(self):
        return self.epoch_average_recall.result()

    def reset_state(self):
        self.epoch_average_recall.reset_state()


class MulticlassWeightedCrossEntropy_1(tf.keras.losses.Loss):
    '''
    Weights are calculated as w_i = (1 - ratio_i), where ratio is the prevalence of that class in the given batch
    '''

    def __init__(self, name="multiclass_weighted_crossent"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # tf.print('BATCH_SIZE: ', tf.shape(y_true)[0])

        class_weights = tf.cast(tf.math.reduce_sum(y_true, axis=0), dtype=tf.int32) / tf.shape(y_true)[0]

        # tf.print('weights0: ', class_weights[0])
        # tf.print('weights1: ', class_weights[1])
        # tf.print('weights2: ', class_weights[2])

        class_weights = tf.cast(class_weights, dtype=tf.float32)

        ce = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)

        # class_ce =[]
        # for idx, w in enumerate(class_weights):
        for i in range(3):
            w = class_weights[i]
            if w != 0:
                # class_ce.append(tf.keras.metrics.categorical_crossentropy(y_true[y_true[:,idx] == 1], y_pred[y_true[:,idx] == 1]))
                # ce[y_true[:, idx] == 1] = w*ce[y_true[:, idx] == 1]
                # tf.print('in loop')
                ce = tf.tensor_scatter_nd_update(ce, tf.where(y_true[:, i] == 1), (1-w)*ce[y_true[:, i] == 1])

        # weighted_ce = tf.linalg.matmul(tf.expand_dims(class_weights[class_weights != 0], axis=0), tf.convert_to_tensor(class_ce, dtype=tf.float64))

        return ce


class MulticlassWeightedCrossEntropy_2(tf.keras.losses.Loss):
    '''
    Same implementation as in https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

    Weights are calculated as w_i = n_samples / (n_classes * n_elements_class_i)
    '''

    def __init__(self, n_classes, name="class_weighted_cross_entropy"):
        super().__init__(name=name)
        self.n_classes = n_classes

    def call(self, y_true, y_pred):

        ce = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)

        for i in range(self.n_classes):
            num_class_i = tf.shape(tf.where(y_true[:, i] == 1))[0]
            # tf.print('num_class_i: ', num_class_i)

            if num_class_i > 0:
                w = tf.shape(y_true)[0] / (self.n_classes * num_class_i)
                w = tf.cast(w, dtype=tf.float32)
                # tf.print('weight: ', w)

                ce = tf.tensor_scatter_nd_update(ce, tf.where(y_true[:, i] == 1), w * ce[y_true[:, i] == 1])
                # tf.print('ce updated')

        return ce


class BinaryF1Score(tf.keras.metrics.Metric):
    '''
    Implementation as in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    https://en.wikipedia.org/wiki/Confusion_matrix

    Is the average of the f1-score of each batch.

    I tested it and gives the same result as sklearn.metrics.balanced_accuracy_score(y_true, y_pred).
    Code to test it:
        y_true1 = np.array([0, 0, 0, 0, 1, 1, 1])
        y_pred1 = np.array([0, 0, 1, 0, 1, 0, 1])
        y_true2 = np.array([0, 1, 0, 0, 0, 0, 1])
        y_pred2 = np.array([0, 0, 1, 0, 1, 0, 1])
        av_bacc = (sklearn.metrics.f1_score(y_true1, y_pred1) + sklearn.metrics.f1_score(y_true2, y_pred2)) / 2
        print(av_bacc)
        y_true1 = tf.convert_to_tensor(y_true1, dtype=tf.float64)
        y_pred1 = tf.convert_to_tensor(y_pred1, dtype=tf.float64)
        y_true2 = tf.convert_to_tensor(y_true2, dtype=tf.float64)
        y_pred2 = tf.convert_to_tensor(y_pred2, dtype=tf.float64)
        BBA = BinaryF1Score()
        BBA.update_state(y_true1, y_pred1)
        BBA.update_state(y_true2, y_pred2)
        print(BBA.result())
    '''

    def __init__(self, name='F1_score', **kwargs):
        super(BinaryF1Score, self).__init__(name=name, **kwargs)
        self.TP = tf.keras.metrics.TruePositives()
        self.FP = tf.keras.metrics.FalsePositives()
        self.FN = tf.keras.metrics.FalseNegatives()
        self.epoch_average_F1 = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):

        # tf.print(y_true.dtype)
        # tf.print((y_pred>0.5).dtype)

        self.TP.reset_state()
        self.FP.reset_state()
        self.FN.reset_state()

        self.TP.update_state(y_true>0.5, y_pred>0.5)
        self.FP.update_state(y_true>0.5, y_pred>0.5)
        self.FN.update_state(y_true>0.5, y_pred>0.5)

        f1_score = 2*self.TP.result() / (2*self.TP.result() + self.FP.result() + self.FN.result())

        self.epoch_average_F1.update_state(f1_score)

    def result(self):
        return self.epoch_average_F1.result()

    def reset_state(self):
        self.epoch_average_F1.reset_state()


class BinaryBalancedAccuracy(tf.keras.metrics.Metric):
    '''
    Gives the average of the balanced accuracy over each batch.

    Same implementation as in https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/metrics/_classification.py#L1933

    I tested it and gives the same result as sklearn.metrics.balanced_accuracy_score(y_true, y_pred).
    Code to test it:
        y_true1 = np.array([0, 0, 0, 0, 1, 1, 1])
        y_pred1 = np.array([0, 0, 1, 0, 1, 0, 1])
        y_true2 = np.array([0, 1, 0, 0, 0, 0, 1])
        y_pred2 = np.array([0, 0, 1, 0, 1, 0, 1])
        av_bacc = (sklearn.metrics.balanced_accuracy_score(y_true1, y_pred1) + sklearn.metrics.balanced_accuracy_score(y_true2, y_pred2))/2
        print(av_bacc)
        y_true1 = tf.convert_to_tensor(y_true1)
        y_pred1 = tf.convert_to_tensor(y_pred1)
        y_true2 = tf.convert_to_tensor(y_true2)
        y_pred2 = tf.convert_to_tensor(y_pred2)
        BBA = BinaryBalancedAccuracy()
        BBA.update_state(y_true1, y_pred1)
        BBA.update_state(y_true2, y_pred2)
        print(BBA.result())
    '''

    def __init__(self, name='binary_balanced_accuracy', **kwargs):
        super(BinaryBalancedAccuracy, self).__init__(name=name, **kwargs)
        self.TP = tf.keras.metrics.TruePositives()
        self.FN = tf.keras.metrics.FalseNegatives()
        self.FP = tf.keras.metrics.FalsePositives()
        self.TN = tf.keras.metrics.TrueNegatives()
        self.epoch_average_accuracy = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):

        # if tf.math.reduce_sum(y_true) == 0:
        #     tf.print('no arts')
        self.TP.reset_state()
        self.FN.reset_state()
        self.FP.reset_state()
        self.TN.reset_state()

        self.TP.update_state(y_true, y_pred)
        self.FN.update_state(y_true, y_pred)
        self.FP.update_state(y_true, y_pred)
        self.TN.update_state(y_true, y_pred)

        # if ((self.TP.result() + self.FN.result())==0) or ((self.TN.result() + self.FP.result())==0):
        #     tf.print('zero')

        sensitivity = self.TP.result() / (self.TP.result() + self.FN.result())
        specificity = self.TN.result() / (self.TN.result() + self.FP.result())

        balanced_accuracy = (sensitivity + specificity)/2

        self.epoch_average_accuracy.update_state(balanced_accuracy)

    def result(self):
        return self.epoch_average_accuracy.result()

    def reset_state(self):
        self.epoch_average_accuracy.reset_state()


class BinaryWeightedCrossEntropy(tf.keras.losses.Loss):
    '''
    Same implementation as in https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

    Weights are calculated as w_i = n_samples / (n_classes * n_elements_class_i)
    '''

    def __init__(self, name="binary_weighted_cross_entropy"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):

        ce = tf.keras.metrics.binary_crossentropy(y_true, y_pred)
        # tf.print('ce: ', ce)

        for i in range(2):
            num_class_i = tf.shape(tf.where(y_true == i))[0]
            # tf.print('num_class_i: ', num_class_i)

            if num_class_i > 0:
                w = tf.shape(y_true)[0] / (2 * num_class_i)
                w = tf.cast(w, dtype=tf.float32)
                # tf.print('weight: ', w)

                indexes = y_true == i

                ce = tf.tensor_scatter_nd_update(ce, tf.expand_dims(tf.where(y_true == i)[:, 0], axis=1), w * ce[indexes[:, 0]])

                # tf.print('ce updated')

        # tf.print('ce_updated: ', ce)
        return ce
