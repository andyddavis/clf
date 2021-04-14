# default options for the input parameters
tpl_installdir="/usr/local/clf_external" # third party library install directory

# read the optional user arguments
while :; do
  case $1 in
    --tpl_dir) # third party library install directory
    if [ "$2" ]; then
      tpl_installdir=$2
      shift
    fi
    ;;
    *) # default case---no more options, break the while loop
    break
  esac
  shift
done

# set (temporary) environment variables for the CLF install
export CLF_TPL_INSTALL_DIR=${tpl_installdir}

# if the install log already exists, remove it
install_log=CLF_PIP_INSTALL_LOG.txt
if [ -f "${install_log}" ]; then
rm ${install_log}
fi

# install using pip
pip3 install . --no-binary none --log ${install_log}

# unset variables for CLF install
unset CLF_TPL_INSTALL_DIR
