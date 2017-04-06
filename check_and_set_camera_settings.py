import subprocess
import time


while True:
	time.sleep(2)
	print subprocess.check_output(['v4l2-ctl', '-C', 'exposure_absolute'])
	print subprocess.check_output(['v4l2-ctl', '-C', 'exposure_auto'])
	if "exposure_absolute: 5\n" not in subprocess.check_output(['v4l2-ctl', '-C', 'exposure_absolute']):
		print "inside if"
		subprocess.call(["v4l2-ctl", "-c", "exposure_auto=1"])
		subprocess.call(["v4l2-ctl", "-c", "exposure_absolute=5"])
