import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change codec if needed
out = cv2.VideoWriter('test_output.avi', fourcc, 10.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    out.write(frame)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
