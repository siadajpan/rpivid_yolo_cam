import cv2
from ultralytics import YOLO
from multiprocessing import Process, Event, Queue
import time
import os
import av
import numpy as np
import sys
import select

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Directory to save frames
SAVE_DIR = "frames"
FPS = 30.0


# Function to ensure the save directory exists
def ensure_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

# Function for reading the stream and setting the event when a frame is availab>
def frame_producer(stream_url, frame_queue, stop_event):
    try:
        container = av.open(stream_url)
    except av.error.ConnectionRefusedError:
        stop_event.set()
        print("Connection refused")
        return

    for frame in container.decode(video=0):
        if stop_event.is_set():
            break

        frame_img = frame.to_ndarray(format="bgr24")

        if not frame_queue.full():
            frame_queue.put(frame_img)
       
        time.sleep(0.03)  # Adjust based on frame rate
    
    container.close()
    

# Function for detecting persons in the frames
def frame_consumer(frame_queue, stop_event, person_event):
    person_detected = False

    while not stop_event.is_set():  # Run until stop_event is triggered
        frame = frame_queue.get()
        print("frame queue elements", frame_queue.qsize())

        # Perform YOLO inference
        results = model(frame)

        # Check if a person is detected in the frame
        for result in results:
            classes = [cls.item() for cls in result.boxes.cls]
            names = np.array([result.names[cls] for cls in classes])
            confidences = np.array([float(conf.item()) for conf in result.boxes.conf])
            print("result", classes, names, confidences)
            
            if len(classes) == 0:
                continue

            person_index = np.where(names == "person")

            if "person" in names and confidences[person_index[0]] > 0.5:
                person_detected = True
                print("Person detected!")
                person_event.set()  # Signal that a person is detected
                break
            else:
                person_detected = False

        if not person_detected:
            person_event.clear()  # Reset the person event if no person detected

            qsize = frame_queue.qsize()
            for _ in range(qsize):
                frame_queue.get()

            print("Emptying the queue:", qsize, "elements")


# Function for saving frames when a person is detected
def frame_saver(frame_queue, person_event, stop_event):
    video_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while not stop_event.is_set():
        out = None
        person_event.wait()  # Wait until a person is detected

        while person_event.is_set():
            frame = frame_queue.get()

            if out is None:
                video_path = os.path.join(SAVE_DIR, f"video_{video_count:04d}.mp4")
                print("frame shape", frame.shape)
                height, width = frame.shape[-2:]
                out = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

            out.write(frame)
            cv2.imwrite(frame_path, frame)  # Save the frame
            print(f"Saved {frame.shape}: {frame_path}")

        out.release()
        video_count += 1


# Main process that starts the producer, consumer, and saver
def main():
    ensure_save_dir()  # Create frames directory if it doesn't exist
    frame_queue = Queue(maxsize=50)  # Shared queue for frames
    stop_event = Event()            # Event to signal when to stop processes
    person_event = Event()          # Event to signal when a person is detected

    # URL of the TCP stream
    stream_url = "tcp://127.0.0.1:8888"

    # Create and start the frame producer, consumer, and saver processes
    producer_process = Process(target=frame_producer, args=(stream_url, frame_queue, stop_event))
    consumer_process = Process(target=frame_consumer, args=(frame_queue, stop_event, person_event))
    saver_process = Process(target=frame_saver, args=(frame_queue, person_event, stop_event))

    producer_process.start()
    consumer_process.start()
    saver_process.start()

    try:
        while not stop_event.is_set():
            # Let the processes run for a while
            keyboard_pressed = select.select([sys.stdin], [], [], 1)[0]
            if keyboard_pressed:
                print("keyboard pressed")
                break
            else:
                print("keyboard not pressed")

    except KeyboardInterrupt:
        print("Keyboard interruption")

    print("Stopping processes...")

    # Set stop event to signal all processes to stop
    stop_event.set()

    # Wait for all processes to finish
    producer_process.join()
    consumer_process.join()
    saver_process.join()

if __name__ == "__main__":
    main()
