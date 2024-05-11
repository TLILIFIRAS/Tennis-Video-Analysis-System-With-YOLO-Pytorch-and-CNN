import cv2

def read_video(video_path):
    """
    Function to read frames from a video file.

    Args:
    - video_path (str): Path to the video file.

    Returns:
    - frames (list): List of frames read from the video.

    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize an empty list to store frames
    frames = []

    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if frame is read successfully
        if not ret:
            break

        # Append the frame to the list
        frames.append(frame)

    # Release the video capture object
    cap.release()

    # Return the list of frames
    return frames

import cv2

def save_video(output_video_frames, output_video_path):
    """
    Function to save frames as a video.

    Args:
    - output_video_frames (list): List of frames to be saved as a video.
    - output_video_path (str): Path where the output video will be saved.

    """

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    # Write frames to the output video
    for frame in output_video_frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    # Print a message indicating that the video has been saved
    print(f"Video saved to {output_video_path}")

# Example usage:
# save_video(output_video_frames, "output_video.mp4")
