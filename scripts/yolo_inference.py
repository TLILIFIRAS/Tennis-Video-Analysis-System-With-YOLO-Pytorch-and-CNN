from ultralytics import YOLO

def detect_objects(video_path, model_type='yolov8x', confidence_threshold=0.2, save_output=True):
    """
    Function to perform object detection on a video using YOLO model.

    Args:
    - video_path (str): Path to the input video file.
    - model_type (str): Type of YOLO model to use (default is 'yolov8x').
    - confidence_threshold (float): Confidence threshold for object detection (default is 0.2).
    - save_output (bool): Whether to save the output predictions (default is True).

    Returns:
    - result (list): List of objects detected in each frame of the video.

    """

    # Create YOLO object detection model
    model = YOLO(model_type)

    # Perform object detection on the video
    result = model.predict(video_path, save=save_output, conf=confidence_threshold)

    # Return the result
    return result

# Example usage:
# result = detect_objects("D:\Computer_Vision_Projects\Tennis Analysis System With YOLO , Pytorch and CNN\Input_Videos\input_video.mp4")
# for box in result[0].boxes:
#     print(box)
