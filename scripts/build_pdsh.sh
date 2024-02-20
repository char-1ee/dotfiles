# Install pdsh with ssh support
cd ${BLOOM_DIR}/github
git clone https://github.com/chaos/pdsh
cd pdsh
./bootstrap
./configure --prefix=${HOME}/.local --with-ssh
make install -j $(nproc)

# Config ssh auth
yes | ssh-keygen -t ecdsa -f ${HOME}/.ssh/id_ecdsa -N "" -vvv
ssh-keygen -y -f ${HOME}/.ssh/id_ecdsa >> ${HOME}/.ssh/authorized_keys