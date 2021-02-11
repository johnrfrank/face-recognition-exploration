# After installing Raspbian on a micro SD card using Raspberry Pi
# Imager, unplug & plug it so that it mounts on /Volumes/boot.

cp wpa_supplicant.conf /Volumes/boot
touch /volumes/boot/ssh


# but this next part doesn't work because ExtFS or microSD is flakey

# Also, by installing the ExtFS for Mac, the main linux volume will
# mount on /Volumes/rootfs, so that we can copy the code into pi
# user's home directory.

#mkdir -p /Volumes/rootfs/home/pi/faces
#cp face-identifier{.py,.service,-setup.sh} /Volumes/rootfs/home/pi/faces/
#chmod a+x /Volumes/rootfs/home/pi/faces/*.{py,sh}
