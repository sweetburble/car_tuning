import cv2 as cv
import numpy as np

# 동영상 파일 경로
video_path = './data/Night Sky Stars.webm'

# 동영상 파일 읽기
cap = cv.VideoCapture(video_path)

# 영상 변환을 위해 원본 동영상의 프레임의 너비와 높이, 초당 프레임 수(fps) 가져오기
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# 만화(Cartoon) 스타일로 변환된 동영상을 저장할 파일 설정
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('Night Sky Stars_cartoon.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    valid, frame = cap.read()
    
    if not valid:
        break

    # 1. 색상 단순화
    # 블러를 사용하여 색상을 단순화하고, 선명한 색상을 유지
    blurred = cv.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. 엣지 검출
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edges = cv.adaptiveThreshold(cv.medianBlur(gray, 7), 255,
                                 cv.ADAPTIVE_THRESH_MEAN_C,
                                 cv.THRESH_BINARY, blockSize=9, C=2)

    # 3. 색상과 엣지 합성
    # 엣지 영역은 유지하되, 나머지 부분은 원본 영상 색상을 보존
    cartoon = cv.bitwise_and(blurred, blurred, mask=edges)

    # 만화 스타일 영상 저장
    out.write(cartoon)

    # 만화 스타일 영상 표시 (실시간 확인을 위해)
    cv.imshow('Cartoon Style Video', cartoon)

    # ESC 키를 누르면 종료
    if cv.waitKey(1) == 27:
        break

# 자원 해제
cap.release()
out.release()
cv.destroyAllWindows()
