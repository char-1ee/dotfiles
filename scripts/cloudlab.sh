# Clone the code
git clone --branch firecracker-v1.4.1-vhive-integration --recurse-submodules https://github.com/char-1ee/firecracker-containerd.git
git clone https://github.com/char-1ee/vHive.git

# Git config
git config --global user.name 'char-1ee'
git config --global user.email xingjianli59@gmail.com

# Install golang 
export GO_VERSION=1.19
wget https://dl.google.com/go/go1.19.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.19.linux-amd64.tar.gz
echo "export PATH=\$PATH:/usr/local/go/bin" >> ~/.bashrc
echo "export GOPATH=\$HOME/go" >> ~/.bashrc
echo "export PATH=\$PATH:\$GOPATH/bin" >> ~/.bashrc
source ~/.bashrc
go --version

# Install docker
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world

# Change fccd/service.go
# Go mod fccd -> remote snapshotter
# Change vHive/vhive_test.go, iface_test.go, makefile, iface.go, manager.go(op)
# log.level(debug)

# Binary
cd firecracker-containerd
make all
cp runtime/containerd-shim-aws-firecracker ~/vHive/bin/
cp firecracker-control/cmd/containerd/firecracker-containerd ~/vHive/bin/
cp firecracker-control/cmd/containerd/firecracker-ctr ~/vHive/bin/

# Comment makefile devtool strip
make firecracker
cp _submodules/firecracker/build/cargo_target/x86_64-unknown-linux-musl/release/jailer ~/vHive/bin/
cp _submodules/firecracker/build/cargo_target/x86_64-unknown-linux-musl/release/firecracker ~/vHive/bin/

make image
# rm ~/vHive/bin/default-rootfs.img
cp tools/image-builder/rootfs.img ~/vHive/bin/default-rootfs.img

cd ~/vHive
touch run.sh
# Write run.sh
# ./scripts/clean_fcctr.sh
# ./scripts/cloudlab/setup_node.sh
# go build -race -v -a ./...
# make debug > output.log 2>&1
# code output.log



# Clone firecracker-containerd
# git clone https://github.com/char-1ee/firecracker-containerd.git
# cd firecracker-containerd
# git branch -r
# git fetch origin firecracker-v1.4.1-vhive-integration:firecracker-v1.4.1-vhive-integration
# git checkout firecracker-v1.4.1-vhive-integration
# make image

# Clone vHive 
# cd ..
# git clone https://github.com/char-1ee/vHive.git
# cd vHive
# git fetch origin firecracker-v1.4.1-vhive-integration:firecracker-v1.4.1-vhive-integration
# git checkout firecracker-v1.4.1-vhive-integration

# Need enable UPF functions in vHive

# Integration test and CRI test
# https://github.com/vhive-serverless/vHive/wiki/TMP:-How-to-run-integration-tests

# ./scripts/cloudlab/setup_node.sh
# go build -race -v -a ./...
# make test-man-bench
# # clean
# ./scripts/clean_fcctr.sh

# ./scripts/cloudlab/setup_node.sh
# # in 3 separate terminals, run
# sudo containerd
# sudo /usr/local/bin/firecracker-containerd --config /etc/firecracker-containerd/config.toml
# source /etc/profile && go build && sudo ./vhive
# # in new termical, create a k8s cluster
# cd vhive
# ./scripts/cluster/create_one_node_cluster.sh
# # wait for the pods to boot up
# watch kubectl get pods -A
# # Setup local registry
# go run ./examples/registry/populate_registry.go -imageFile ./examples/registry/images.txt
# # deploy functions
# kn service apply helloworld -f ./configs/knative_workloads/helloworld.yaml
# kn service apply helloworldlocal -f ./configs/knative_workloads/helloworld_local.yaml
# kn service apply helloworldserial -f ./configs/knative_workloads/helloworldSerial.yaml
# kn service apply pyaes -f ./configs/knative_workloads/pyaes.yaml
# # run tests
# cd cri
# go test ./ -v -race -cover
# # clean 
# ./scripts/github_runner/clean_cri_runner.sh

