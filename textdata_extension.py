import tensorflow as tf
from tensorflow_examples.lite.model_customization.core.data_util.text_dataloader import TextClassifierDataLoader
from tensorflow_examples.lite.model_customization.core.model_export_format import ModelExportFormat
import tensorflow_examples.lite.model_customization.core.task.text_classifier as text_classifier

@classmethod
def from_panda(cls, df, class_labels=None, num_classes=2, shuffle=True):
    """Text analysis for text classification load text with labels.

    Assume the text data of the same label are in the same subdirectory. each
    file is one text.

    Args:
      filename: Name of the file.
      class_labels: Class labels that should be considered. Name of the
        subdirectory not in `class_labels` will be ignored. If None, all the
        subdirectories will be considered.
      shuffle: boolean, if shuffle, random shuffle data.

    Returns:
      TextDataset containing images, labels and other related info.
    """
    pd_len = df.shape[0]
    if shuffle:
      #shuffle panda object
      pass

    # Gets label and its index.
    if class_labels:
      label_names = sorted(class_labels)
    else:
      label_names = ['0','1']
    ## neccesary?
    ## new code
    text_ds = tf.data.Dataset.from_tensor_slices( tf.cast(df['review'].values, tf.string) )
    label_ds = tf.data.Dataset.from_tensor_slices( tf.cast(df['sentiment'].values, tf.int64) )
    # print(type(text_ds), type(label_ds))
    text_label_ds = tf.data.Dataset.zip( (text_ds, label_ds) )
    #text_label_ds = tf.data.Dataset.from_tensor_slices( (tf.cast(df['review'].values, tf.string), tf.cast(df['rating'].values, tf.int32)) )
    return TextClassifierDataLoader(text_label_ds, pd_len,
                                    num_classes, label_names)

