import cv2 as cv 
eyes_area=cv.CascadeClassifier('C:/Users/anura/Downloads/haarcascade_eye.xml')
face_area=cv.CascadeClassifier('C:/Users/anura/Downloads/haarcascade_frontalface_default.xml')

cap=cv.VideoCapture(0)

while True:
    ret, frame =cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY )
    
    #face detection
    face=face_area.detectMultiScale(gray, 1.3,5)
    
    for (x, y, w, h) in face:
      gray_roi=gray[y:y+h, x:x+w]
      #color_roi=frame[y:y+h, x:x+w]
      cv.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 4)
    
    
    #eyes detection
      eyes=eyes_area.detectMultiScale(gray_roi)
      for(ex,ey,ew,eh) in eyes:
          eye_region=gray_roi[ey:ey+eh, ex:ex+eh]
          
          #dark circle check
          
          avg_intensity=eye_region.mean()
          #if avg_intensity < 500: # it will detect deeply
          if avg_intensity < 40:  # it will detect upar upar se
              
              print("dark circle detected")
          else:
            print(" no dark circles")
              
    cv.imshow("live face detection", frame)
    
    if cv.waitKey(1) & 0xFF== ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()