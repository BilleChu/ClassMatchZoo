from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
import keras.backend as K
import numpy as np
import time

class Checkpoint(ModelCheckpoint):
    def __init__(self):
        super(Checkpoint, self).__init__(filepath="./model/weights.{epoch:03d}-{val_acc:.4f}.hdf5", 
                                        monitor='val_acc', 
                                        verbose=1, 
                                        save_best_only=True, 
                                        mode='auto')


class StaticHistory(Callback):
    def __init__(self, test_data, test_label, categories):
        local_time = time.localtime()
        self.logfile =  "log/logs_" + str(local_time.tm_mon) +\
                        "_" + str(local_time.tm_mday) +\
                        "_" + str(local_time.tm_hour) +\
                        "_" + str(local_time.tm_min)
        self.test_data = test_data
        self.test_label = test_label
        self.categories = categories
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
        self.lr = []
    def on_epoch_end(self, epoch, logs={}):
        output = self.model.predict(self.test_data, batch_size=128)
        preds  = np.argmax(output, axis=-1)
        labels = np.argmax(self.test_label, axis=-1)
        print (len(preds), len(labels))
        with open(self.logfile, "a") as fwrite:
            fwrite.write("-"*25 + "*"*5 + "-"*25 + "\n")
            fwrite.write("epoch  -->" + str(epoch) + "\n")
            fwrite.write("loss   -->" + str(logs.get("loss")) + "\n")
            fwrite.write("acc    -->" + str(logs.get("acc"))+ "\n")
            fwrite.write("-"*25 + "-"*5 + "-"*25 + "\n")
            fwrite.write(classification_report(labels, preds, target_names=self.categories))
            fwrite.write("-"*25 + "-"*5 + "-"*25 + "\n")
            np.savetxt(fwrite, confusion_matrix(y_true=labels, y_pred=preds), fmt="%d")
            fwrite.write("-"*25 + "*"*5 + "-"*25 + "\n")

class LrateScheduler(LearningRateScheduler):
    def __init__(self):
        super(LrateScheduler, self).__init__(schedule=self.step_decay)

    def step_decay(self, epoch):
        if epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(self.model.optimizer.lr)