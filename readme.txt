https://github.com/FRC-1294/frc2016vision



------------ Change the raspberry pi hostname ---------------
sudo nano /etc/hostname
Replace raspberrypi with raspberrypi-1294
Press CTRL+X to close the editor; agree to overwrite the existing file and save it.
sudo nano /etc/hosts
Leave all of the entries alone except for the very last entry labeled 127.0.1.1 with the hostname “raspberrypi“. This is the only line you want to edit. Replace “raspberrypi” with "raspberrypi-1294".
Press CTRL+X to close the editor; agree to overwrite the existing file and save it.



--------- Configure raspberry pi to not launch X --------- 
sudo raspi-config
3 Boot Options
B1 Console
Finish, Yes to reboot


--------- Run vision on raspberry pi boot -----------
Copy ./vision.service from this project to /etc/systemd/system on the RPi
sudo systemctl enable vision.service
