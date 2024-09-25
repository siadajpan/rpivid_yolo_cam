# camera stream (on raspi 192.168.129.14)
systemctl status rpicam-vid
on 127.0.0.1:8888

# to build:
sudo docker build -t yolo-infer .

# to run:
 sudo docker run -d --restart unless-stopped --network host -v /home/karol/rpivid_yolo_cam:/app -it yolo-infer

# to get the videos:
 cp -r karol@192.168.129.14:/home/karol/rpivid_yolo_cam/frames .