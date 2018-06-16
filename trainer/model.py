import tensorflow as tf
from tensorflow.python.framework import ops

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.constant([[1.0]])
  labels = tf.constant([0])

  # Dense Layer
  dense = tf.layers.dense(inputs=input_layer, units=1, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  
  print(mode)
  print(features)
  print(labels)
  print(tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions, train_op=train_op, export_outputs={
    'regression': tf.estimator.export.RegressionOutput(tf.constant(1.0, dtype=tf.float32)),
    'classification': tf.estimator.export.ClassificationOutput(scores=tf.constant(1.0, dtype=tf.float32)),
    'predict': tf.estimator.export.PredictOutput(tf.constant(1.0, dtype=tf.float32)),
    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(tf.constant(1.0, dtype=tf.float32))
  })

def estimator(config):
    return tf.estimator.Estimator(model_fn=cnn_model_fn, config=config)
