# Coupled local functions (CLF)

Our goal is to find <img src="https://render.githubusercontent.com/render/math?math=u"> such that <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}(u)=f"> for a differential operator <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}"> and forcing function <img src="https://render.githubusercontent.com/render/math?math=f">. 

Our approach represents <img src="https://render.githubusercontent.com/render/math?math=\hat{u} \approx u"> using <em>local polynomials</em> (e.g., see [Davis et al. (2020)](https://arxiv.org/abs/2006.00032), [Kohler (2002)](https://link.springer.com/article/10.1023/A:1022427805425), and [Stone (1977)](https://www.jstor.org/stable/2958783?casa_token=HSIT0xXYt_4AAAAA%3AlVXC5N7urbFzbX3rVp5gtcXLUH8sLGU3s8vxGa0rO7I1VCVnQDOaOnHAW8XshlOn_aeQk0Ai8XOq7GXz5Nc1Br2Ll6Og8PFgLnx-Kk1vUUMyXn9g0Z9P&seq=1#metadata_info_tab_contents)). In particular, we solve 

<img src="https://render.githubusercontent.com/render/math?math=\argmin_{\hat{u} \in \mathcal{P}} \int_{\Omega} \frac{1}{2} \| \mathcal{L}(\hat{u}) - f \|^2 d \pi(x)">, 

where 
<img src="https://render.githubusercontent.com/render/math?math=\pi"> is a probability density function that we use to define the residual error and <img src="https://render.githubusercontent.com/render/math?math=\mathcal{R}"> is a regularization. We sample points <img src="https://render.githubusercontent.com/render/math?math=x_i \sim \pi"> and define a <img src="https://render.githubusercontent.com/render/math?math=p"> degree polynomial in a local region. We solve a regression problem at collocation points <img src="https://render.githubusercontent.com/render/math?math=y_{ij} \sim q(\cdot \vert x_i)"> in the local neighborhood.

# Building, installing, and running the tests

After cloning this repository, build and install using

```
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/install/dir/
make -jN
make install
cd ..
```

Run the tests using 

```
./build/run-tests
```

Developer note: the Python interface is outdated and not currently functional. We will update this when we have finalized the mathematical formulation.

# Documentation

To view the documentation open the file `documentation/html/index.html` after cloning this repository or run `make doc` after running `cmake` to build the most up-to-date documentation.

[//]: # (This may not currently be functional, but to install using <tt>pip3</tt> run the command <tt>./install.sh --tpl_dir ~/Software/install/clf-pip/clf_external</tt>.)
[//]: # (cmake .. -DCLF_BOOST_DIR= -DCLF_EIGEN3_DIR= -DCLF_GTEST_DIR= -DCLF_MUQ_DIR= -DCLF_NLOPT_DIR=)
