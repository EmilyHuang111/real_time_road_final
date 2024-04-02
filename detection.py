import io
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk   
from datetime import datetime
import logging
import pytz
import threading
import time 
import pickle
import glob
from compass  import *

# Root Menu for user registration and login
root = tk.Tk()
root.title("User Login")

# Set up logging with US/Eastern timezone
eastern = pytz.timezone('US/Eastern')

#Set up global variables for video streaming control
stop_video_raw = threading.Event()
stop_video_processed = threading.Event()

# Initialize webcam
video_file_path = "./video/IMG_0412.mp4"

webcam = cv.VideoCapture(video_file_path)

webcam_lock = threading.Lock()

# Updat the lable for log file
def update_label(image, label):
    photo = ImageTk.PhotoImage(image=image)
    label.config(image=photo)
    label.image = photo  # Keep a reference

# Set up logging
logging.basicConfig(
    filename='activity_log.txt',
    level=logging.INFO,
    format='%(message)s',
)

# Function for the log information including date and time and message
def log_activity(message):
  current_time = datetime.now(eastern).strftime('%Y-%m-%d %I:%M:%S %p')
  logging.info(f"{current_time} - {message}")

# Global variable to control video replay
stop_video_replay = threading.Event()

curvature_values = []  # Stores recent curvature values for moving average calculation
n_values_for_average = 10  # Number of values to calculate the moving average, adjust as needed

def undistort_img():
    # Prepare object points 0,0,0 ... 8,5,0
    obj_pts = np.zeros((6*9,3), np.float32)
    obj_pts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    objpoints = []  # Stores all object points
    imgpoints = []  # Stores all image points

    # Get directory for all calibration images
    images = glob.glob('camera_cal/*.jpg')
    img_size = None

    for indx, fname in enumerate(images):
        img = cv.imread(fname)
        if img is None:
            continue  # Skip if image not loaded
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])  # Define img_size using the first successfully loaded image

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)

        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)

    if img_size is None:
        raise Exception("No valid images found in 'camera_cal/*.jpg'.")

    # Calibrate camera using the object points and image points
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv.undistort(img, mtx, dist, None, mtx)
    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump( dist_pickle, open('camera_cal/cal_pickle.p', 'wb') )

undistort_img()

def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv.undistort(img, mtx, dist, None, mtx)
    
    return dst

def pipeline(img, s_thresh=(15, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS).astype(np.float64)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv.Sobel(l_channel, cv.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv.warpPerspective(img, M, dst_size)
    return warped

def inv_perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.85),(0.58,0.85),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv.warpPerspective(img, M, dst_size)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=9, margin=150, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c 
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    # Set height of windows
    window_height = int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    if lefty.size > 0 and leftx.size > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])
        
        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])
          
    else:
    # Handle the case where no lane pixels were detected
    # For example, use a default polynomial, skip this iteration, or log a warning
     print("No left lane pixels detected.")

    # Fit a second order polynomial to each
    if len(righty) > 0 and len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])
        
        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])
        
    else:
    # Handle the case where no lane pixels were detected
    # For example, use a default polynomial, skip this iteration, or log a warning
     print("No left lane pixels detected.")

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def get_curve(img, leftx, rightx, ploty):
    y_eval = np.max(ploty)
    
    # Assume typical lane line length in meters and a lane's width in meters to calculate meters per pixel
    lane_length_m = 30.5  # Typical length of lane line in meters
    lane_width_m = 3.7    # Typical width of a lane in meters
    
    # Calculate meters per pixel in y and x dimensions based on actual image size
    ym_per_pix = lane_length_m / img.shape[0]  # meters per pixel in y dimension
    xm_per_pix = lane_width_m / img.shape[1]   # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    epsilon = 1e-6
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / (np.absolute(2 * left_fit_cr[0])+epsilon)
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / (np.absolute(2 * right_fit_cr[0])+epsilon)

    # Calculate the direction of the curve
    left_curve_dir = np.sign(2 * left_fit_cr[0])
    right_curve_dir = np.sign(2 * right_fit_cr[0])

    # Apply the sign convention to the curvature values
    left_curverad *= left_curve_dir
    right_curverad *= right_curve_dir

    # Calculate car position and lane center position
    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10  # Center offset in meters

    return (left_curverad, right_curverad, center)

def draw_lanes(img, left_fit, right_fit,ploty):
    #ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    # Calculate points for the left and right lanes
    left_lane = np.array([np.transpose(np.vstack([left_fit, ploty]))], dtype=np.int32)
    right_lane = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))], dtype=np.int32)

    # Draw the left lane line
    for i in range(len(ploty)-1):
        cv.line(color_img, (int(left_lane[0][i][0]), int(left_lane[0][i][1])), 
                (int(left_lane[0][i+1][0]), int(left_lane[0][i+1][1])), (0, 255, 0), thickness=30)
    
    # Draw the right lane line
    for i in range(len(ploty)-1):
        cv.line(color_img, (int(right_lane[0][i][0]), int(right_lane[0][i][1])), 
                (int(right_lane[0][i+1][0]), int(right_lane[0][i+1][1])), (0, 255, 0), thickness=30)

    # Calculate and draw the centerline with increased thickness
    center_fit = (left_fit + right_fit) / 2
    for i in range(len(ploty)-1):
        cv.line(color_img, (int(center_fit[i]), int(ploty[i])), (int(center_fit[i+1]), int(ploty[i+1])), (255, 0, 0), thickness=30)
    
    # Apply inverse perspective warp to get the result back to the original perspective
    # Assuming inv_perspective_warp is defined elsewhere
    inv_perspective = inv_perspective_warp(color_img, 
                                           dst_size=(img.shape[1], img.shape[0]),
                                           src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),  # Update these points
                                           dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]))  # Update these points

    # Overlay the lane markings on the original image with 50% transparency
    result = cv.addWeighted(img, 1, inv_perspective, 1., 0)
    return result

def overlay_image(background, overlay, position=(0, 0)):
    x, y = position
    h, w = overlay.shape[:2]  # Height and width of the overlay image

    # Ensure overlay_region is the first three channels, and alpha_channel is normalized
    overlay_region = overlay[:, :, :3]  # RGB channels of the overlay
    alpha_channel = overlay[:, :, 3] / 255.0  # Normalize the alpha channel, ensuring it is a float

    # Calculate the ending x and y positions for the overlay region
    end_x = x + w
    end_y = y + h

    # Ensure the overlay fits within the background dimensions
    if end_x > background.shape[1] or end_y > background.shape[0]:
        overlay_region = overlay_region[:min(h, background.shape[0] - y), :min(w, background.shape[1] - x)]
        alpha_channel = alpha_channel[:min(h, background.shape[0] - y), :min(w, background.shape[1] - x)]
    
    # Extract the region of interest from the background
    background_region = background[y:y+h, x:x+w]

    # Adjust alpha_channel shape to be broadcastable over the color channels
    alpha_channel = alpha_channel[..., np.newaxis]  # Add a new axis to match with RGB channels

    # Blend the overlay with the background using the alpha channel
    blended_region = (1.0 - alpha_channel) * background_region + alpha_channel * overlay_region

    # Place the blended region back into the background image
    background[y:y+h, x:x+w] = blended_region.astype(np.uint8)

    return background

def vid_pipeline(img, current_frame, fps,userName):
    global running_avg, index, curvature_values, n_values_for_average

    # Define times and positions for images
    image_times = [(0, 22), (22, 45), (46, 56), (57, 79), (80, 90), (91, 94), (76, 80)]
    image_positions = [(100, 100), (100, 100), (100, 100), (100, 100), (100, 100), (100, 100)]

    # Load overlay images
    overlay_images_paths = [
        "./straight.png",
        "./left.png",
        "./straight.png",
        "./straight.png",
        "./right.png",
        "./straight.png"
    ]
    overlay_images = [cv.imread(path, cv.IMREAD_UNCHANGED) for path in overlay_images_paths]

    play_time = current_frame / fps
    
    # Process the frame
    img_ = pipeline(img)
    img_ = perspective_warp(img_)
    out_img, curves, lanes, ploty = sliding_window(img_, draw_windows=False)
    curverad = get_curve(img, curves[0], curves[1], ploty)
    lane_curve = np.mean([curverad[0], curverad[1]])
    lane_curve_ft = lane_curve * 3.28084  # Convert from meters to feet
    
    curvature_values.append(lane_curve_ft)
    if len(curvature_values) > n_values_for_average:
        curvature_values.pop(0)
    
    avg_curvature = sum(curvature_values) / len(curvature_values)
    img = draw_lanes(img, curves[0], curves[1], ploty)
    
    compassPosition = (img.shape[1] // 2, 700)
    img = draw_compass(img, avg_curvature, compassPosition)

    # Overlay images based on time
    for i, ((add_time, remove_time), position) in enumerate(zip(image_times, image_positions)):
        factor = 4.7
        if add_time / factor <= play_time < remove_time / factor:
            overlay_img = overlay_images[i]
            img = overlay_image(img, overlay_img, position)
            # Ensure logging occurs only every 10th frame
            if current_frame % 100 == 0:  # Checks if the current frame number is divisible by 100
                # Check the image path for logging
                if overlay_images_paths[i].endswith("left.png"):
                    log_activity(f"{userName} turns left.")             
            if current_frame % 50 == 0:  # Checks if the current frame number is divisible by 50
                # Check the image path for logging
                if overlay_images_paths[i].endswith("right.png"):
                    log_activity(f"{userName} turns right.")   
    return img

def safe_webcam_read(cap, max_attempts=5):
    attempts = 0
    while attempts < max_attempts:
        ret, frame = cap.read()
        if ret:
            return True, frame
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts} failed to read from webcam.")
        time.sleep(0.1)  # Brief pause before retrying
    return False, None  # Indicate failure after all attempts

def load_video_raw(currentState, userName, frame, stop_event):
    global webcam, webcam_lock
    currentState.config(text="Current State: Camera loaded")
    video_label2 = tk.Label(frame)
    video_label2.grid(row=2, column=0, columnspan=2)
    
    if not webcam.isOpened():
        currentState.config(text="Current State: No Connection")
        return  # Exit the function if the webcam is not opened
    fps = 25
    while not stop_event.is_set():
        with webcam_lock:
            ret, frame = safe_webcam_read(webcam)
        if ret:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            rgb_frame = cv.resize(rgb_frame, (256, 256))
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)
            video_label2.config(image=photo)
            video_label2.image = photo  # Keep a reference
            root.after(0, update_label, Image.fromarray(rgb_frame), video_label2)
            time.sleep(1 / fps)  # Delay frame loading
        else:
            # If the video ends (no frame returned), rewind it to the start
            with webcam_lock:
                webcam.set(cv.CAP_PROP_POS_FRAMES, 0)

def load_video_processed(currentState, userName, frame, stop_event):
    global webcam, webcam_lock
    currentState.config(text="Current State: Overlay Loaded")
    video_label1 = tk.Label(frame)
    video_label1.grid(row=2, column=0, columnspan=2)

    fps = 25
    total_frames = 2350  # Total frames in the video

    if not webcam.isOpened():
        currentState.config(text="Current State: No Connection")
        return

    frame_count = 0  # Initialize frame count

    while not stop_event.is_set():
        with webcam_lock:
            ret, original_frame = safe_webcam_read(webcam)
            current_frame_pos = webcam.get(cv.CAP_PROP_POS_FRAMES)  # Get the current frame position

            if not ret or current_frame_pos >= total_frames:
                # If the video ends or fails to read, rewind to the start
                webcam.set(cv.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0  # Reset frame count for the new loop
                continue  # Skip the current iteration and try reading again

        processed_frame = vid_pipeline(original_frame, frame_count, fps,userName)
        rgb_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
        rgb_frame = cv.resize(rgb_frame, (256, 256))
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image=image)
        video_label1.config(image=photo)
        video_label1.image = photo  # Keep a reference
        root.after(0, update_label, Image.fromarray(rgb_frame), video_label1)
        
        frame_count += 1  # Increment frame count
        time.sleep(1 / fps)  # Attempt to maintain the frame rate
