import tensorflow as tf

def config(job_dir):
    return tf.estimator.RunConfig(
        tf_random_seed=19830610,
        log_step_count_steps=1000,
        save_checkpoints_secs=120,  # change if you want to change frequency of saving checkpoints
        keep_checkpoint_max=3,
        model_dir=job_dir
    )
