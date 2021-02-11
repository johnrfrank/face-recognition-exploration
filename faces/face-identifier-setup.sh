

# use python2 version, because mod_python is compiled against py2

sudo apt-get update
sudo apt-get install libqt4-test libqtgui4 libjasper-dev libatlas-base-dev libhdf5-dev \
     libatlas-base-dev python3-yaml python3-matplotlib -y

pip3 install opencv-contrib-python==4.1.0.25

# enable python to bind to low-numbered (privileged) ports
sudo setcap CAP_NET_BIND_SERVICE=+eip /usr/bin/python3.7

sudo cp /home/pi/faces/face-identifier.service /etc/systemd/system
sudo systemctl start face-identifier.service
sudo systemctl enable face-identifier.service

echo "start_x=1" | sudo tee -a /boot/config.txt
echo "gpu_mem=1" | sudo tee -a /boot/config.txt

sudo reboot
