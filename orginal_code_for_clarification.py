import cv2, os, datetime, time, subprocess
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
Picamera2.set_logging(Picamera2.ERROR)
import os
os.environ["LIBCAMERA_LOG_LEVELS"] = "3"

# region - Constants
# region - Calibration
def calibration(cr_path):
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (1296,972)}, lores={"size": (640, 480)}, display="lores")
    picam2.configure(camera_config)
    # picam2.start_preview(Preview.QTGL)
    picam2.start()
    co_ordinates = [[960,540],[30,30],[30,540],[30,1050],[1890,30],[1890,540],[1890,1050],[960,30],[960,1050]]
    places = ['center','top_left','center_left', 'bottom_left', 'top_right','center_right', 'bottom_right','center_top','center_bottom']
    for i,p in zip(co_ordinates,places):
        canvas = cv2.imread(black_image_path)
        canvas = cv2.circle(canvas,(round(i[0]),round(i[1])),30,[102,65,249],-1)
        window_name = 'Calibration'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name,canvas)
        cv2.waitKey(500)
        for shrink in [28,25,22,18,15,12,10,8,6]:
            canvas = cv2.imread(black_image_path)
            canvas = cv2.circle(canvas,(round(i[0]),round(i[1])),shrink,[102,65,249],-1)
            window_name = 'Calibration'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name,canvas)
            cv2.waitKey(100)
            
        for shrink in [6,8,10]:
            canvas = cv2.imread(black_image_path)
            canvas = cv2.circle(canvas,(round(i[0]),round(i[1])),shrink,[89,223,252],-1)
            window_name = 'Calibration'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name,canvas)
            cv2.waitKey(50)
        picam2.capture_file(cr_path+p+'.jpg')
        for shrink in [8,6]:
            canvas = cv2.imread(black_image_path)
            canvas = cv2.circle(canvas,(round(i[0]),round(i[1])),shrink,[89,223,252],-1)
            window_name = 'Calibration'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name,canvas)
            cv2.waitKey(50)
    cv2.destroyAllWindows()
    picam2.stop()
    picam2.close()
    
def drift_calc(cr_path):
    # picam2 = Picamera2()
    # video_config = picam2.create_video_configuration(main={"size": (1296,972)}, lores={"size": (640,480)}, display="lores")
    # picam2.configure(video_config)
    # picam2.start()
    window_name = 'Look at the Point'
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter(cr_path+'drift.mp4', fourcc, 10.0, (1296,972))
    # time.sleep(10)
    # points = [(40,40),(480,270),(1440,810),(1880,1040),(40,1040),(480,810),(1440,270),(1880,40),(40,540),(480,540),(960,540),(1440,540),(1880,540)]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    points = [(480,270),(1440,270),(480,810),(1440,810),(960,540)]
    original = cv2.imread(black_image_path)
    picam2 = Picamera2()
    picam2.video_configuration.main.size = (1296, 972)
    picam2.video_configuration.controls.FrameRate = 10.0
    encoder = H264Encoder(bitrate=10000000)
    output = FfmpegOutput(cr_path+'drift.mp4')
    picam2.start_recording(encoder, output)
    for i in points:
        # black_image = cv2.imread(black_image_path)
        # black_image = cv2.circle(black_image,i,20,[0,0,255],-1)
        # cv2.imshow(window_name,black_image)
        # cv2.waitKey(5000)
        for repeat in range(3):
            for shrink in range(3,22,1):
                canvas = original.copy()
                canvas = cv2.circle(canvas,(i[0],i[1]),shrink,[0,0,255],-1)
                cv2.imshow(window_name,canvas)
                cv2.waitKey(40)
            for shrink in range(22,3,-1):
                canvas = original.copy()
                canvas = cv2.circle(canvas,(i[0],i[1]),shrink,[0,0,249],-1)
                cv2.imshow(window_name,canvas)
                cv2.waitKey(40)
        # time.sleep(5)
        # count = 0
        # end = time.time()+5
        # while time.time()<end:
        #     count+=1
        #     frame = picam2.capture_array()
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     out.write(frame)
        #     time.sleep(0.07)
        # print('frames',count,count/5)
    picam2.stop_recording()
    cv2.destroyAllWindows()
    picam2.close()
    
def record(cr_path):

    # video_config = picam2.create_video_configuration(main={"size": (1296,972)}, lores={"size": (640,480)}, display="lores",controls={"FrameRate": 30}) # , controls={"FrameDurationLimits": (33333, 33333)}
    # picam2.configure(video_config)
    # picam2.video_configuration.controls.FrameRate = 30.0
    # picam2.start()
    import random
    x = list(range(1,16))
    points = [[165,200], [560,200], [945,200], [1311,200], [1713,200], 
			[165,555], [560,555], [945,555], [1311,555], [1685,555],
			[135,925], [530,925], [915,925], [1280,925], [1685,925]]
    nums = list(range(1,16))
    result = []
    for i in range(4):
        choice = random.choice(x)
        result.append(str(choice))
        x.remove(choice)
    window_name = 'Free-view'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    picam2 = Picamera2()
    picam2.video_configuration.main.size = (1296, 972)
    picam2.video_configuration.controls.FrameRate = 10.0
    encoder = H264Encoder(bitrate=10000000)
    output = FfmpegOutput(cr_path+'user-feed.mp4')
    picam2.start_recording(encoder, output)
    for point in result:
        black_image = cv2.imread('/home/shakra/Downloads/Test_image_just_lines.png')
        black_image = cv2.putText(black_image, str(nums[int(point)-1]), points[int(point)-1], cv2.FONT_HERSHEY_SIMPLEX,  2, (0,0,255), 3, cv2.LINE_AA) 
        cv2.imshow(window_name,black_image)
        cv2.waitKey(5000)
        # # time.sleep(5)
        # count = 0
        # end = time.time()+5
        # while time.time()<end:
        #     count+=1
            # frame = picam2.capture_array()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # out.write(frame)
            # time.sleep(0.07)
        # print('frames',count,count/5)
           
    picam2.stop_recording()
    # out.release()
    cv2.destroyAllWindows()
    picam2.close()
    print('Data Collection Done')
    result = [int(x) for x in result]
    subprocess.run(['python3.10', '/home/shakra/Downloads/post_processing_drift&test_editing.py', cr_path, str(result)])
    print('Processing Finished')
    
black_image_path = r'/home/shakra/Downloads/solid-black.jpg'  # Image should be of 1920x1080
cr_folder = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
cr_path = './output/' + cr_folder + '/'
if not os.path.exists('./output'):
    os.mkdir('./output')
os.mkdir(cr_path)
#endregion - Constants
time.sleep(1)

calibration(cr_path)
drift_calc(cr_path)
record(cr_path)

# user_name = input('Name : ')
# print("""Types of test:
#       1. Without Head movement.
#       2. With Head Movement.
#       3. Front and back movement. ( Distance from screen ).
#       4. With glasses.
#       5. All Done.""")
# while 1:
#     test_type = input("Type of Test ? (Enter no): ")
#     if test_type=='5':
#         break
#     cr_folder = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")+'_'+user_name+'_'+test_type
#     cr_path = './output/' + cr_folder + '/'
#     if not os.path.exists('./output'):
#         os.mkdir('./output')
#     os.mkdir(cr_path)
#     #endregion - Constants
#     time.sleep(1)

#     calibration(cr_path)
#     drift_calc(cr_path)
#     record(cr_path)