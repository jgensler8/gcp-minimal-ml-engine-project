import tensorflow as tf

def generate():
    return (1.0, 1.0)

def train_input_fn():
    print("training")
    dataset = tf.data.Dataset.from_generator(generate, (tf.float64, tf.float64))
    iterator = dataset.make_one_shot_iterator()
    # features, target = iterator.get_next()
    # return features, target
    return tf.constant([ [[1],[1]], [[1], [2]] ]), [1]
    
def eval_input_fn():
    print("evaluating")
    # dataset = tf.data.Dataset.from_generator(generate, (tf.float64, tf.float64))
    # iterator = dataset.make_one_shot_iterator()
    # features, target = iterator.get_next()
    # return features, target
    return [2, 3], [1]

def json_serving_function():
    inputs = {
        'one': tf.constant([1]),
        'zero': tf.constant([0])
    }

    return tf.estimator.export.ServingInputReceiver(
        features={
            'one': tf.constant([1]),
            'zero': tf.constant([0])
        },
        receiver_tensors=inputs,
        # receiver_tensors_alternatives=inputs
    )
