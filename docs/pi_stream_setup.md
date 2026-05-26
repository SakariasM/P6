PI STREAM SETUP
===============
Pi IP:       192.168.10.3
Pi user:     sw6
Pi password: jFTYQvI88pCnsHXXWHgZ
Pi project:  ~/Project/Prototype/P6/


PREREQUISITES (run once on new PC)
-----------------------------------
sudo apt-get install ffmpeg sshpass openssh-server
sudo systemctl start ssh


ASSIGN STATIC IP TO ETHERNET (run each session)
-------------------------------------------------
sudo ip addr add 192.168.10.5/24 dev enx803f5d09ca75


CHECK CONNECTION
----------------
ping 192.168.10.3


ONE-TIME SSH KEY SETUP (Pi → PC, run once)
-------------------------------------------
Allows the Pi to SCP files to this PC without a password.

# 1. Generate key on Pi
sshpass -p 'jFTYQvI88pCnsHXXWHgZ' ssh -o StrictHostKeyChecking=no sw6@192.168.10.3 \
  "ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519"

# 2. Copy Pi's public key to this PC
PI_KEY=$(sshpass -p 'jFTYQvI88pCnsHXXWHgZ' ssh -o StrictHostKeyChecking=no sw6@192.168.10.3 "cat ~/.ssh/id_ed25519.pub")
echo "$PI_KEY" >> ~/.ssh/authorized_keys

# 3. Test (should print OK with no password prompt)
sshpass -p 'jFTYQvI88pCnsHXXWHgZ' ssh sw6@192.168.10.3 \
  "ssh -o StrictHostKeyChecking=no flemming@192.168.10.5 'echo OK'"


SET PI SOURCE TO UDP STREAM (run once)
----------------------------------------
sshpass -p 'jFTYQvI88pCnsHXXWHgZ' ssh -o StrictHostKeyChecking=no sw6@192.168.10.3 \
  "sed -i 's/SOURCE         = 0/SOURCE         = \"udp:\/\/@:1234\"/' ~/Project/Prototype/P6/live_mask.py"


RUN BENCHMARK (standard flow)
------------------------------
# Terminal 1 — SSH into Pi and run the benchmark script
ssh sw6@192.168.10.3
cd ~/Project/Prototype/P6
./run_benchmark.sh

# Terminal 2 — stream video from PC to Pi
ffmpeg -re -i /home/flemming/Downloads/test_footage_40s.mp4 -vf scale=960:540 -f mpegts udp://192.168.10.3:1234

When ffmpeg finishes the video ends, the Pi model stops automatically,
and the pred_mask file is sent to:
  /home/flemming/Documents/GitHub/P6/data/preds/pred_mask_{model}_{WxH}.mp4


STREAM ON LOOP (degradation / thermal test)
--------------------------------------------
ffmpeg -stream_loop -1 -re -i /home/flemming/Downloads/test_footage_40s.mp4 -vf scale=960:540 -f mpegts udp://192.168.10.3:1234


REVERT PI BACK TO WEBCAM
-------------------------
sshpass -p 'jFTYQvI88pCnsHXXWHgZ' ssh sw6@192.168.10.3 \
  "sed -i 's/SOURCE         = \"udp:\/\/@:1234\"/SOURCE         = 0/' ~/Project/Prototype/P6/live_mask.py"

