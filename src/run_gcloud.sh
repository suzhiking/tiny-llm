# Make sure the Compute Engine API is enabled (one-time)
gcloud services enable compute.googleapis.com

# Pick ONE zone (Toronto region has -a, -b, -c)
export ZONE=northamerica-northeast2-b
export VM_NAME=bpe-runner
export MACHINE_TYPE=n2-standard-8

# (Optional) also set default region/zone for your session
gcloud config set compute/region northamerica-northeast2
gcloud config set compute/zone   $ZONE
gcloud services enable compute.googleapis.com

# Create the VM in that single zone; use ≥200GB disk for better I/O
gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced
