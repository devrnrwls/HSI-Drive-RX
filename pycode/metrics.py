import os
import gc
import tensorflow as tf
import json

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
gc.collect()

src_path = os.path.dirname(os.path.abspath(__file__))
dataset_config_path = os.path.join(src_path, 'Dataset.json')
with open(dataset_config_path, 'r') as f:
    data = json.load(f)
    aux = data["versionDASIP"]
f.close()

#OJO, esto obliga a que, de manera externa, los pesos que se necesiten se coloquen en class_weight y global_weight
#Para eso, también es necesario escoger entre las versiones DASIP (1.1) y 2.0 de la base de datos
class_weight = aux["weights"]
class_weight = class_weight[1:(len(class_weight)+1)]
global_weight = aux["global_weights"]
global_weight = global_weight[1:(len(global_weight)+1)]

#Seguro que hay una manera más elegante de hacer estas clases.
#Me he basado en lo que pone en https://www.tensorflow.org/guide/keras/train_and_evaluate
#Todas tienen el nombre Sparse por delante porque los GT están dados en formato sparse y no one-hot

#Esta métrica que se actualiza cada iteración me indica cuántos TP se han predicho en esa iteración.
#Esta métrica nunca va a ir ponderada porque no es infomartiva --> no debe ir en la casilla weighted_metrics
class CustomSparseCategoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name="cat_true_pos", **kwargs):
        super(CustomSparseCategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # The state of the metric will be reset at the start of each ITERATION.
        self.true_positives.assign(0.0)
        y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), shape=(-1, 1)), "int8")
        y_true = tf.cast(tf.reshape(y_true, shape=(-1, 1)), "int8")

        #Como no me interesa la clase 0, solo calculo los TP para el resto de clases.
        values = tf.cast((y_pred == y_true) & (y_pred != 0), "float32")
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


#Esta métrica que se actualiza cada iteración me indica cuántos píxeles del GT tienen etiqueta distinta de 0 (están etiquetados).
#Comparando la salida con la dada por CategoricalTruePositives me permite tener más control sobre la fracción de píxeles que se etiqueta correctamente.
#Esta métrica tampoco tiene sentido que vaya ponderada.
class CustomSparseNumberOfLabelledPixels(tf.keras.metrics.Metric):
    def __init__(self, name="num_lab_pixels", **kwargs):
        super(CustomSparseNumberOfLabelledPixels, self).__init__(name=name, **kwargs)
        self.labelled_pixels = self.add_weight(name="nlp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # The state of the metric will be reset at the start of each ITERATION.
        self.labelled_pixels.assign(0.0)
        y_true = tf.cast(tf.reshape(y_true, shape=(-1, 1)), "int8")
        values = tf.cast(y_true != 0, "float32")
        self.labelled_pixels.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.labelled_pixels

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.labelled_pixels.assign(0.0)


#Esta métrica que se actualiza cada iteración indica cuál es el IoU
#Si la incluyo en weighted metrics, me va a dar el weighted IoU mientras que si la incluyo en metrics me va a dar el global IoU
class CustomSparseCategoricalIoU(tf.keras.metrics.Metric):
    def __init__(self, name="cat_iou", **kwargs):
        super(CustomSparseCategoricalIoU, self).__init__(name=name, **kwargs)
        self.sparse_iou = self.add_weight(name="sci", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # The state of the metric will be reset at the start of each ITERATION.
        self.sparse_iou.assign(0.0)
        #Tiene que haber una manera más elegante de pasar el número de clases (este valor ya incluye la clase 0.)
        num_classes = y_pred.shape[-1]
        y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), shape=(-1, 1)), "int8")
        y_true = tf.cast(tf.reshape(y_true, shape=(-1, 1)), "int8")
        classIou = []

        #Como no me interesa la clase 0, solo calculo los TP para el resto de clases.
        for c in range(1, num_classes):
            TP = tf.reduce_sum(tf.cast((y_true != 0) & (y_true == c) & (y_pred == c), "float32"))
            FN = tf.reduce_sum(tf.cast((y_true != 0) & (y_true == c) & (y_pred != c), "float32"))
            FP = tf.reduce_sum(tf.cast((y_true != 0) & (y_true != c) & (y_pred == c), "float32"))
            IoU = TP / (TP + FP + FN)
            classIou.append(IoU)

        #sample_weight no es None cuando se coloca la métrica en weighted_metrics ya que internamente se le pasa la variable sample_weight.
        if sample_weight is not None:
            weighted_IoU = tf.multiply(classIou, class_weight)
            self.sparse_iou.assign_add(tf.reduce_sum(weighted_IoU))
        else:
            global_IoU = tf.multiply(classIou, global_weight)
            self.sparse_iou.assign_add(tf.reduce_sum(global_IoU))

    def result(self):
        return self.sparse_iou

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sparse_iou.assign(0.0)


#Esta métrica que se actualiza cada iteración indica cuál es la precision
#Si la incluyo en weighted metrics, me va a dar el weighted precision mientras que si la incluyo en metrics me va a dar el global precision
class CustomSparseCategoricalPrecision(tf.keras.metrics.Metric):
    def __init__(self, name="cat_pre", **kwargs):
        super(CustomSparseCategoricalPrecision, self).__init__(name=name, **kwargs)
        self.sparse_pre = self.add_weight(name="scp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # The state of the metric will be reset at the start of each ITERATION.
        self.sparse_pre.assign(0.0)
        num_classes = y_pred.shape[-1]
        y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), shape=(-1, 1)), "int8")
        y_true = tf.cast(tf.reshape(y_true, shape=(-1, 1)), "int8")
        classPre = []

        #Como no me interesa la clase 0, solo calculo los TP para el resto de clases.
        for c in range(1, num_classes):
            TP = tf.reduce_sum(tf.cast((y_true != 0) & (y_true == c) & (y_pred == c), "float32"))
            FP = tf.reduce_sum(tf.cast((y_true != 0) & (y_true != c) & (y_pred == c), "float32"))
            Pre = TP / (TP + FP)
            classPre.append(Pre)

        #sample_weight no es None cuando se coloca la métrica en weighted_metrics ya que internamente se le pasa la variable sample_weight.
        if sample_weight is not None:
            weighted_Pre = tf.multiply(classPre, class_weight)
            self.sparse_pre.assign_add(tf.reduce_sum(weighted_Pre))
        else:
            global_Pre = tf.multiply(classPre, global_weight)
            self.sparse_pre.assign_add(tf.reduce_sum(global_Pre))

    def result(self):
        return self.sparse_pre

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sparse_pre.assign(0.0)


#Esta métrica que se actualiza cada iteración indica cuál es la accuracy
#Si la incluyo en weighted metrics, me va a dar el weighted accuracy mientras que si la incluyo en metrics me va a dar el global accuracy
class CustomSparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="cat_acc", **kwargs):
        super(CustomSparseCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.sparse_acc = self.add_weight(name="sca", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # The state of the metric will be reset at the start of each ITERATION.
        self.sparse_acc.assign(0.0)
        num_classes = y_pred.shape[-1]
        y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), shape=(-1, 1)), "int8")
        y_true = tf.cast(tf.reshape(y_true, shape=(-1, 1)), "int8")
        classAcc = []

        #Como no me interesa la clase 0, solo calculo los TP para el resto de clases.
        for c in range(1, num_classes):
            TP = tf.reduce_sum(tf.cast((y_true != 0) & (y_true == c) & (y_pred == c), "float32"))
            FN = tf.reduce_sum(tf.cast((y_true != 0) & (y_true == c) & (y_pred != c), "float32"))
            Acc = TP / (TP + FN)
            classAcc.append(Acc)

        #sample_weight no es None cuando se coloca la métrica en weighted_metrics ya que internamente se le pasa la variable sample_weight.
        if sample_weight is not None:
            weighted_Pre = tf.multiply(classAcc, class_weight)
            self.sparse_acc.assign_add(tf.reduce_sum(weighted_Pre))
        else:
            global_Pre = tf.multiply(classAcc, global_weight)
            self.sparse_acc.assign_add(tf.reduce_sum(global_Pre))

    def result(self):
        return self.sparse_acc

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sparse_acc.assign(0.0)
