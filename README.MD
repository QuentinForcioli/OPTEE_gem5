# OPTEE Trusted firmware
This is demo for OPTEE on gem5.

# Dependencies
The compiler I'm using:
* aarch64-linux-gnu-gcc
* arm-linux-gnueabihf-gcc 
* aarch64-linux-gnu-gcc

You may also need some dependencies from OPTEE (https://optee.readthedocs.io/en/latest/building/prerequisites.html) though everything is not needed for this demo.

You can modify the compilers by directly changing the make file in :
* ``application/lib/optee_tf-a``
* ``application/template/linux_optee_on_aarch64``
  
# Running the demo
To run you have to provide a gem5 path with ARM ``gem5.opt`` already built.\
``
make GEM5_PATH=../gem5/ 
``