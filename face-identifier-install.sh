
echo "First, we login using password 'raspberry' and run the setup script."
ssh-keygen -R "raspberrypi.local"
scp -i ~/.ssh/airjrf6-id_rsa-2020-07-23 -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -r faces pi@raspberrypi.local:faces
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no pi@raspberrypi.local "sudo chown -R pi: faces ; chmod a+x faces/*.{py,sh} ; /home/pi/faces/face-identifier-setup.sh"
until $(curl --output - --silent --fail http://raspberrypi.local/ | grep "Update Model" > /dev/null); do
    printf '.'
    sleep 5
done

echo "Now we login again using password 'raspberry' and remove the WiFi password."
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no pi@raspberrypi.local "sudo rm -f /etc/wpa_supplicant/wpa_supplicant.conf ; sudo shutdown -h now"


