#/bin/bash
v4l2-ctl --set-ctrl=exposure_auto=1
v4l2-ctl --set-ctrl=exposure_absolute=5
python /home/pi/2017_vision_tracking/the_real_vision_detector.py
