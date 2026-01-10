# Ray Cluster Configuration for Earthlink

# Head node
ray start \
  --head \
  --node-ip-address=127.0.0.1 \
  --port=6380 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-cpus=4 \
  --num-gpus=1 \
  --memory=8589934592 \
  --object-store-memory=2147483648 \
  --metrics-export-port=8080 \
  --include-dashboard=true

# Worker nodes (run on separate machines)
# ray start \
#   --address='<head-node-ip>:6380' \
#   --num-cpus=8 \
#   --num-gpus=1 \
#   --memory=17179869184 \
#   --object-store-memory=4294967296
