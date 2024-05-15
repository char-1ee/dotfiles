wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3

sudo mkdir /etc/modulefiles
cd /etc/modulefiles

# #%Module1.0
# proc ModulesHelp { } {
#     puts stderr "Activates the Miniconda3 environment"
# }
# module-whatis "Loads the Miniconda3 environment"

# # Set the path to where Miniconda is installed
# set root /opt/miniconda3

# prepend-path PATH $root/bin

export MODULEPATH=$MODULEPATH:/etc/modulefiles

