import cv2
import face_recognition
import numpy as np
import datetime
from datetime import timedelta

# set desired runtime in minutes
min_run = 8
start_time = datetime.datetime.now()
end_time = start_time + timedelta(minutes = min_run)

# video capture 0 will use default camera
#video_capture = cv2.VideoCapture(0)

# run ./start_stream in ~/bin/ on raspberry ip to start video stream and get valid hostname
#video_capture = cv2.VideoCapture('http://192.168.1.10:8080/?action=stream')

video_capture = cv2.VideoCapture('http://129.118.162.82:8080/?action=stream')

# load reference images for facial recognition
afam_image = face_recognition.load_image_file("afam.jpg")
afam_face_encoding = face_recognition.face_encodings(afam_image)[0]

jake_image = face_recognition.load_image_file("jake.jpg")
jake_face_encoding = face_recognition.face_encodings(jake_image)[0]

sunny_image = face_recognition.load_image_file("sunny.jpg")
sunny_face_encoding = face_recognition.face_encodings(sunny_image)[0]

pham_image = face_recognition.load_image_file("pham.jpg")
pham_face_encoding = face_recognition.face_encodings(pham_image)[0]

bipana_image = face_recognition.load_image_file("bipana.jpg")
bipana_face_encoding = face_recognition.face_encodings(bipana_image)[0]

cameron_image = face_recognition.load_image_file("cameron.jpg")
cameron_face_encoding = face_recognition.face_encodings(cameron_image)[0]

jaturong_image = face_recognition.load_image_file("jaturong.jpg")
jaturong_face_encoding = face_recognition.face_encodings(jaturong_image)[0]

himel_image = face_recognition.load_image_file("himel.jpg")
himel_face_encoding = face_recognition.face_encodings(himel_image)[0]

linh_image = face_recognition.load_image_file("linh.jpg")
linh_face_encoding = face_recognition.face_encodings(linh_image)[0]


# array of known face encodings
known_face_encodings = [
    afam_face_encoding,
    jake_face_encoding,
    sunny_face_encoding,
    pham_face_encoding,
    bipana_face_encoding,
    cameron_face_encoding,
    jaturong_face_encoding,
    himel_face_encoding,
    linh_face_encoding
]


# array of known face names
known_face_names = [
    "Afam",
    "Jake",
    "Sunny",
    "Pham",
    "Bipana",
    "Cameron",
    "Jaturong",
    "Himel",
    "Linh"
]


# initialize variables
face_locations = []
face_encodings = []
face_names = []
present_names = []
absent_names = []

# used to process every second frame
process_this_frame = True

# get current date 
date = datetime.datetime.now()

# create a string for date with year month and day
year = str(date.year)
month = str(date.month)
day = str(date.day)
date1 = (year +'-'+ month +'-'+ day)

# create a file unique to the day
f = open(date1+'_record'+'.txt', 'w+')
f.write('Arrival:\n')
f.close()

while True:
    # single frame is retrieved
    ret, frame = video_capture.read()

    # resize frame of video to .25 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # process every second frame 
    if process_this_frame:
        # find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # see if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # if a match was found in known_face_encodings, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # check each line in report file for matched name
                with open(date1+'_record'+'.txt') as file:
                    datafile = file.readlines()
                found = False
                for line in datafile:
                    if name in line:
                        found = True
                        
                # if name not yet in file
                if not found:
                    # append name to present_names[]
                    present_names.append(name)

                    # write name and time of arrival to file 
                    f = open(date1+'_record'+'.txt', 'a')
                    currentDT = datetime.datetime.now()
                    f.write(name + '\t' + str(currentDT) + '\n')
                    f.close()

            face_names.append(name)
            
    # alternate frames to process (only needed when a high framerate is used)
    #process_this_frame = not process_this_frame

    # display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # scale back up face locations since the frame we detected in was scaled to .25 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # display the resulting image
    cv2.imshow('Video', frame)

    # hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    curr_time = datetime.datetime.now()
    elap_time = curr_time - start_time

    print(elap_time)
    if curr_time >= end_time:
        break

# open record file 
f = open(date1+'_record'+'.txt', 'a')

# sort and write present names to file
present_names.sort()
f.write('\nPresent:\n')
for name in present_names:
    f.write(name+'\n')

# find absend names
for name1 in known_face_names:
    found = False
    for name2 in present_names:
        if name1 == name2:
            found = True
    if not found:
        absent_names.append(name1)
        
# sort and write absent names to file
absent_names.sort()
f.write('\nAbsent:\n')
for name in absent_names:
    f.write(name+'\n')
    
f.close()

# release handle camera
video_capture.release()
cv2.destroyAllWindows()
