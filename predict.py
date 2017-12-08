from darkflow.net.build import TFNet
import cv2
import time
 
options = {"model": "cfg/tiny-yolo-voc-2c.cfg", "load": -1, "threshold": 0.1}

tfnet = TFNet(options)
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(True):
    start = time.time()
    ret, frame = cap.read()
    result = tfnet.return_predict(frame)
    for a in result:
        label=a['label']
        confidence=a['confidence']
        porc='{0:.0f}%'.format(confidence*100)
        if confidence>=0.1:
            xt=a['topleft']['x']
            yt=a['topleft']['y']
            xb=a['bottomright']['x']
            yb=a['bottomright']['y']
            cv2.rectangle(frame,(xt,yt),(xb,yb),(0,255,0),2)
            cv2.putText(frame,'{} {}'.format(label,porc),(xt-10,yt-10),0,0.3,(0,255,0))
    end = time.time()
    seconds=end-start
    fps=1/seconds
    cv2.putText(frame,'{} FPS'.format(fps),(100,100),0,0.3,(0,255,0))
    #out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#imgcv = cv2.imread("./sample_img/dog.jpg")
#result = tfnet.return_predict(imgcv)
#print(result)
