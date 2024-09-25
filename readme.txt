# camera stream
systemctl status rpicam-vid
on 127.0.0.1:8888

# to build:
sudo docker build -t yolo-infer .

# to run:
sudo docker run --rm --network host -v /home/karol/yolo_infer:/app -it yolo-infer
