export REGION=northamerica-northeast2
export ZONE="northamerica-northeast2-b northamerica-northeast2-c"
gcloud config set compute/region $REGION
gcloud config set compute/zone   $ZONE

export BUCKET=gs://cs336-gavin-bucket

# Adjust paths if yours differ
gsutil cp ~/Courses/CS336/assignment1-basics/data/corpus/TinyStoriesV2-GPT4-train.txt \
  $BUCKET/corpus/

gsutil cp -r ~/Courses/CS336/assignment1-basics/data/tokenizer_data/tinystories_bpe_train \
  $BUCKET/tokenizer_data/

export VM_NAME=bpe-runner
export MACHINE_TYPE=n2-standard-32    # good perf/$; try n2-standard-16 if cheaper

gcloud compute instances create $VM_NAME \
  --machine-type=$MACHINE_TYPE \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=100GB

