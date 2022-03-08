// To install using <tt>pip3</tt> run the command <tt>./install.sh --tpl_dir ~/Software/install/clf-pip/clf_external</tt>.

[//]: # (This may be the most platform independent comment)

cmake .. \
    -DCLF_BOOST_DIR= \
    -DCLF_EIGEN3_DIR= \
    -DCLF_GTEST_DIR= \
    -DCLF_MUQ_DIR= \
    -DCLF_NLOPT_DIR=
