import tensorflow as tf

import model
import input
import util
import argparse

def main():
    ARGS = args_parser.parse_args()
    print "job_dir: {}".format(ARGS.job_dir)
    config = util.config(ARGS.job_dir)
    
    estimator = model.estimator(config)
    
    train_spec = tf.estimator.TrainSpec(
        input.train_input_fn,
        max_steps=100
    )

    exporter = tf.estimator.FinalExporter(
        'estimator',
        input.json_serving_function,
        as_text=False  # change to true if you want to export the model as readable text
    )

    eval_spec = tf.estimator.EvalSpec(
        input.eval_input_fn,
        exporters=[exporter],
        name='estimator-eval',
        steps=100
    )
    
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )
    
args_parser = argparse.ArgumentParser()
args_parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=True
)
args_parser.add_argument(
    '--train-files',
    help='GCS or local paths to training data',
    nargs='+',
    required=True
)
args_parser.add_argument(
    '--eval-files',
    help='GCS or local paths to evaluation data',
    nargs='+',
    required=True
)

if __name__ == '__main__':
    main()
