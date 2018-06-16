VIRTUALENV_DIR=./env
PIP=${VIRTUALENV_DIR}/bin/pip
ACTIVATE=source ${VIRTUALENV_DIR}/bin/activate

# Python + Environment

virtualenv:
	virtualenv ${VIRTUALENV_DIR}

install: virtualenv
	${PIP} install -e .

# TensorFlow

MODEL_DIR=./output
TRAIN_DATA=./train
EVAL_DATA=./eval

TRAINER_PACKAGE=trainer
TRAINER_MAIN=${TRAINER_PACKAGE}.task

train_local:
	bash -c '${ACTIVATE} && gcloud ml-engine local train \
    --module-name ${TRAINER_MAIN} \
    --package-path ${TRAINER_PACKAGE} \
    --job-dir ${MODEL_DIR} \
    -- \
    --train-files ${TRAIN_DATA} \
    --eval-files ${EVAL_DATA}'

# --train-steps 1000 \
# --eval-steps 100'

BUCKET_NAME=tftraining

upload_train_eval_data:
	echo "would upload training data"
	echo gsutil cp ${TRAIN_DATA} gs://${BUCKET_NAME}/train
	echo gsutil cp ${EVAL_DATA} gs://${BUCKET_NAME}/eval
	
# JOB_NAME=${BUCKET_NAME}_$(shell date +%s)
JOB_NAME=${BUCKET_NAME}_2
BUCKET_JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}
REGION=us-central1
RUNTIME_VERSION=1.5

train_job:
	gcloud ml-engine jobs submit training ${JOB_NAME} \
    --job-dir ${BUCKET_JOB_DIR} \
    --runtime-version ${RUNTIME_VERSION} \
    --module-name ${TRAINER_MAIN} \
    --package-path ${TRAINER_PACKAGE} \
    --region ${REGION} \
    -- \
    --train-files ${TRAIN_DATA} \
    --eval-files ${EVAL_DATA}

MODEL_NAME=helloworld_model

create_model:
	gcloud ml-engine models create ${MODEL_NAME} --regions=${REGION}

MODEL_BINARIES=gs://${BUCKET_NAME}/${JOB_NAME}/export/estimator/1529119938

MODEL_VERSION=v1

create_model_version:
	gcloud ml-engine versions create ${MODEL_VERSION} \
	--model ${MODEL_NAME} \
	--origin ${MODEL_BINARIES} \
	--runtime-version ${RUNTIME_VERSION}

JSON_INSTANCES=./json_instances.jsonl

test_model_version:
	gcloud ml-engine predict \
	  --model ${MODEL_NAME} \
	  --version ${MODEL_VERSION} \
	  --json-instances ${JSON_INSTANCES}
