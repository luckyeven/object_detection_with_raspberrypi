"""
Raspberry Pi Camera Image Capture

Author: Shifeng Song
Date: Nev 13, 2021

"""
import time
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

# image Settings
res_width = 96                     # width of camera
res_height = 96                    # height of camera
rotation = 0                       # Camera rotation (0,90,180,or 270)
draw_fps = False                   # Display fps on screen
save_path = "../images/num02/"      # Save images to directory
file_num = 0                       # Starting point for filename
file_suffix = ".png"               # image Extension
precountdown = 2                   # Seconds before staring countdown
countdown =  5                     # Secounds to count down from

# Initial framerate value
fps = 0

##########################################################
# Functions

def file_exists(filepath):
    """
    Returns true if file exists, false otherwise
    """
    try:
        f = open(filepath, 'r')
        exists = True
        f.close()
    except:
        exists = False
    return exists


def get_filepath():
    """
    Returns the next available full path to image file
    """

    global file_num

    # Loop through possible file numbers to see if that file already exists
    filepath = save_path + str(file_num) + file_suffix
    while file_exists(filepath):
        file_num += 1
        filepath = save_path + str(file_num) + file_suffix

    return filepath

################################################################################
# Main

# Figure out the name of the output image filename
filepath = get_filepath()

# Start the camera
with PiCamera() as camera:

    # Configure camera settings
    camera.resolution = (res_width, res_height)
    camera.rotation = rotation
    
    # Container for our frames
    raw_capture = PiRGBArray(camera, size=(res_width, res_height))
    
    # Initial countdown timestamp
    countdown_timestamp = cv2.getTickCount()

    # Continuously capture frames (this is our while loop)
    for frame in camera.capture_continuous(raw_capture, 
                                            format='bgr', 
                                            use_video_port=True):
                                            
        # Get timestamp for calculating actual framerate
        timestamp = cv2.getTickCount()
        
        # Get Numpy array that represents the image
        img = frame.array
       
        # Each second, decrement countdown
        if (timestamp - countdown_timestamp) / cv2.getTickFrequency() > 1.0:
            countdown_timestamp = cv2.getTickCount()
            countdown -= 1
            
            # When countdown reaches 0, break out of loop to save image
            if countdown <= 0:
                countdown = 0
                break
                
        
        # Draw countdown on screen
        cv2.putText(img,
                    str(countdown),
                    (int(round(res_width / 2) - 5),
                        int(round(res_height / 2))),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255))
                    
        # Draw framerate on frame
        if draw_fps:
            cv2.putText(img, 
                        "FPS: " + str(round(fps, 2)), 
                        (0, 12),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255))
        
        # Show the frame
        cv2.imshow("Frame", img)
        
        # Clear the stream to prepare for next frame
        raw_capture.truncate(0)
        
        # Calculate framrate
        frame_time = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()
        fps = 1 / frame_time
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Capture image

    camera.capture(filepath)
    print("Image saved to:", filepath)
 

# Clean up
cv2.destroyAllWindows()
