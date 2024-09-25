# Use the Ultralytics YOLO image as the base
FROM ultralytics/ultralytics:latest-arm64

# Set the working directory inside the container
WORKDIR /app

RUN mkdir frames
RUN apt install -y vim nano
RUN pip install av

# Set the default command to run the YOLO script
CMD ["python", "yolo_infer.py"]
