import face_recognition
import cv2
import sys
import os
import re 
import serial
import twilio
from twilio.rest import Client

STRANGER_FRAME_LIMIT = 100
FRIENDLY_FRAME_LIMIT = 10
MATCH_THRESHOLD = 0.5

# return dictionary of names and faces of pictures in curr dir
def getDictionaryOfFaces ():

	faceDict = {};
	pictureFiles = [];

	for file in os.listdir(os.getcwd()):
		if (file.endswith(".png")) or (file.endswith(".jpg")):
			print("reading " + file)
			pictureFiles.append(file)

	itr = 0
	for file in pictureFiles:
		knownImage = face_recognition.load_image_file(file)
		knownFaceEncoding = face_recognition.face_encodings(knownImage)[0]
		name = re.sub('\.jpg$', '', file)
		name = re.sub('\.png$', '', name)
		faceDict[itr] = [name, knownFaceEncoding]
		itr = itr + 1
				
	return faceDict 

# from video image, builds a histogram of faces
# unlocks door if FRIENDLY_FRAME_LIMIT is reached
def buildFaceDictionary (faceDict, faceNames, serConn):

	for name in faceNames:
		if name in faceDict:
			faceDict[name] += 1
		else:
			faceDict[name] = 1

		if faceDict[name] == FRIENDLY_FRAME_LIMIT:
			faceDict = {}
			unlockDoor(serConn)

def unlockDoor (serConn): 
	print ("unlocking the door")
	#serConn.write('a')
	
def sendStrangerTextMessage ():
	client = Client("ACa646a19da1675504bd8dcd6347991801", "39301628a478ff69b10bf162c062cd70")

# change the "from_" number to your Twilio number and the "to" number
# to the phone number you signed up for Twilio with, or upgrade your
# account to send SMS to any phone number
#client.api.account.messages.create(to="+18585987176", 
#                           from_="+16193134984", 
#                           body="Warning, someone at the door - Hi from Andrew.")

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# main
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
faceDictionary = getDictionaryOfFaces();

# Initialize some variables
face_locations = []   #the locations in video frame
face_encodings = []   #face encoding in video frame
face_names = []       #holds the names, loose correlation
process_this_frame = True
noMatchItr = 0        #consecutive iterator for strangers
matchedPersonDict = {}#histogram of recog faces
#ser = serial.Serial('/dev/tty.usbmodem1411')
#ser.baudrate = 9600
ser = 3
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            # See if the face is a match for the known face(s)
	    gotMatch = 0
	    for knownPerson in faceDictionary:

		knownPersonEncoding = faceDictionary[knownPerson][1]
   		match = face_recognition.compare_faces([knownPersonEncoding], face_encoding, MATCH_THRESHOLD)

            	if match[0]:
			#stop comparing to other known encodings
			gotMatch = 1
                	name = faceDictionary[knownPerson][0] 
            		face_names.append(name)
			break

	    if not gotMatch:
		face_names.append("Unknown")

    process_this_frame = not process_this_frame

    #determine action based on the video frame and face_names
    foundPerson = False   #indicates if recognized person at door 

    for face in face_names:
       if face != "Unknown":
          foundPerson = True
	  break

    if foundPerson:
	# reset itr and check faceDict
	noMatchItr = 0
	buildFaceDictionary(matchedPersonDict, face_names, ser)
  
    else:
	# inc itr, reset faceDict, check if send text
	if face_names:
	#if there are people
	        noMatchItr += 1
		if matchedPersonDict:
			matchedPersonDict = {}
	
		if noMatchItr == STRANGER_FRAME_LIMIT:
			noMatchItr = 0
			#send text message
			print("Text message someone unknown at the door")
			sendStrangerTextMessage()
	else :
	#if no one is there, reset
		noMatchItr = 0

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
	if (name == "Unknown") :
        	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        	cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	else :
        	cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        	cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()