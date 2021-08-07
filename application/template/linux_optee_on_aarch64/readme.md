# Linux on aarch64
This script automatically compile a program that works on linux using everything inside the ``src`` folder, then it launched it automatically on aarch64-linux using gem5.
It uses the classical ``example/fs.py`` that you can find inside the gem5 config directory.



# Dependencies
This example uses ``aarch64-none-linux-gnu-gcc`` for compiling the target binary.

This example also uses ``guestmount`` and ``guestfish`` to build image and mounted them without requiring root privileges. 

Finally, This example uses headless-trusty-ubuntu image that you can find on the gem5 website. This image can be downloaded [here](https://www.gem5.org/documentation/general_docs/fullsystem/guest_binaries) and then decompressed inside the ``common/system/arm/disks`` folder.

For ``example/fs.py`` to work, you also need to compile the bootloaders and to put them inside the common system directory (inside ``common/system/arm/binaries``).
## Building the bootloader
To build the bootloader, you have to go inside the gem5 system folder specifically ``gem5/system/arm/bootloader/arm64`` and use make (you may need to change the bootloader makefile to use by using ``export CROSS_COMPILE=aarch64-none-elf-``), you may also need the 32bits ones inside ``gem5/system/arm/bootloader/arm`` using make again to build them (you may need to replace this time the compiler used in the makefile by using ``export CROSS_COMPILE=arm-linux-gnueabihf-``)

They will produce the bootloaders that you will need to copy to ``common/system/arm/binaries``: 
    * ``boot_emm.arm``
    * ``boot_emm.arm64``
    * ``boot_v2.arm64``
    * ``boot.arm``
    * ``boot.arm64``

Finally, you will need to have a compiled linux kernel called ``vmlinux`` inside ``common/system/arm/binaries`.
# Building 
You can use the default make commands to build this example :
    
    make
# Cleanup
To clean the repository you can use :

    make clean

# Launching the example
You can automatically launch the example using :

    make gem5

This will automatically try to take a checkpoint after linux booting to launch the built binary from this point onward.
This process can take time for the first boot.

If you change the architecture and or you face problem with the checkpoint. You should delete the m5out directory:

    make clean_m5out

# Contact
> Quentin Forciol : <quentin.forcioli@telecom-paris.fr>