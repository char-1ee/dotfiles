#! /bin/bash

sudo apt-get update

# chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

# docker
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt install docker.io
sudo snap install docker
docker --version

# tableplus
wget -qO - http://deb.tableplus.com/apt.tableplus.com.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://deb.tableplus.com/debian/20 tableplus main"
sudo apt update
sudo apt install tableplus

# golang
sudo snap install go --classic
go version

# gomock
go install github.com/golang/mock/mockgen@v1.6.0

# sqlc
sudo snap install sqlc

# migrate
curl -L https://packagecloud.io/golang-migrate/migrate/gpgkey | apt-key add -
echo "deb https://packagecloud.io/golang-migrate/migrate/ubuntu/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/migrate.list
apt-get update
apt-get install -y migrate

# DB docs
npm install -g dbdocs
dbdocs login

# DBML CLI
npm install -g @dbml/cli
dbml2sql --version