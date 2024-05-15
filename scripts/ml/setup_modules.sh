#!/bin/bash

# Function to create an example modulefile
create_modulefile() {
    local modulefile_path=$1
    local binary_path=$2
    local library_path=$3

    mkdir -p $(dirname $modulefile_path)
    cat > $modulefile_path << EOF
#%Module1.0
prepend-path PATH $binary_path
prepend-path LD_LIBRARY_PATH $library_path
EOF
}

# Main installation and setup function
setup_modules() {
    local os_name=$(grep '^ID=' /etc/os-release | sed -e 's/ID=//g' | tr -d '"')

    # Install Environment Modules based on the OS
    if [[ "$os_name" == "ubuntu" || "$os_name" == "debian" ]]; then
        sudo apt-get update
        sudo apt-get install -y environment-modules
    elif [[ "$os_name" == "centos" || "$os_name" == "rhel" ]]; then
        sudo yum install -y environment-modules
    else
        echo "Unsupported operating system."
        exit 1
    fi

    # Source the module environment initialization script
    echo "source /etc/profile.d/modules.sh" >> $HOME/.bashrc

    # Create a directory for custom modulefiles
    local custom_module_dir="/etc/modulefiles"
    mkdir -p $custom_module_dir

    # Example modulefile for Python
    local example_modulefile="$custom_module_dir/python"
    create_modulefile $example_modulefile "/opt/python/bin" "/opt/python/lib"

    # Print usage examples
    echo "Environment Modules is installed."
    echo "Run 'module load python' to load the example Python module."
    echo "Run 'module list' to see loaded modules."
    echo "Run 'module unload python' to unload the Python module."
    echo "Add 'source /etc/profile.d/modules.sh' to other shell init scripts if necessary."
}

# Execute the main function
setup_modules
