import cv2
import numpy as np

t_array = []
x_array = []
y_array = []

trackers = ["CSRT (Channel and Spatial Reliability Tracker)", "KCF (Kernelized Correlation Filters) Tracker",
            "MIL (Multiple Instance Learning) Tracker", "DaSiamRPN", "Nano Tracker", "Vit Tracker"]

optical_flow_methods = ["Gunnar-Farneback Optical Flow Method", "Lucas-Kanade Optical Flow Method"]


def draw_point(frame, x, y):
    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


def draw_bbox(frame, bbox):
    x, y, w, h = bbox
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)


def general_tracker(first_frame, bbox, cap, out, method):
    cv2.namedWindow(method, cv2.WINDOW_NORMAL)
    if method == trackers[0]:
        tracker = cv2.TrackerCSRT.create()
    elif method == trackers[1]:
        tracker = cv2.TrackerKCF.create()
    elif method == trackers[2]:
        tracker = cv2.TrackerMIL.create()
    elif method == trackers[3]:
        params = cv2.TrackerDaSiamRPN.Params()
        params.kernel_cls1 = 'bin/dasiamrpn_kernel_cls1.onnx'
        params.kernel_r1 = 'bin/dasiamrpn_kernel_r1.onnx'
        params.model = 'bin/dasiamrpn_model.onnx'
        tracker = cv2.TrackerDaSiamRPN.create(params)
    elif method == trackers[4]:
        params = cv2.TrackerNano.Params()
        params.backbone = 'bin/nanotrack_backbone_sim.onnx'
        params.neckhead = 'bin/nanotrack_head_sim.onnx'
        tracker = cv2.TrackerNano.create(params)
    elif method == trackers[5]:
        params = cv2.TrackerVit.Params()
        params.net = 'bin/vitTracker.onnx'
        tracker = cv2.TrackerVit.create(params)

    tracker.init(first_frame, bbox)

    t = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            ret, bbox = tracker.update(frame)
        except Exception as e:
            print(e)
            break

        if ret:
            x = (bbox[0] + bbox[2] / 2)
            y = (bbox[1] + bbox[3] / 2)
            x_array.append(x)
            y_array.append(y)
            draw_point(frame, int(x), int(y))
            draw_bbox(frame, bbox)
            t_array.append(t)
            t += 1
        else:
            cv2.putText(frame, "Tracking Failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)

        cv2.imshow(method, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    return t_array, x_array, y_array


def optical_flow_tracker(first_frame, point, cap, out, method):
    cv2.namedWindow(method, cv2.WINDOW_NORMAL)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_point = point
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    t = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if method == optical_flow_methods[0]:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None,
                                                0.5, 3, 15, 4, 5, 1.2, 0)

            dx, dy = flow[int(prev_point[0, 1]), int(prev_point[0, 0])].astype(np.float32)
            prev_point[0, 0] += dx
            prev_point[0, 1] += dy
            x, y = prev_point.ravel()


        elif method == optical_flow_methods[1]:
            next_point, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_point, None, **lk_params)
            x, y = next_point.ravel()
            prev_gray = gray_frame.copy()
            prev_point = next_point.copy()

        draw_point(frame, int(x), int(y))
        cv2.imshow(method, frame)
        t_array.append(t)
        x_array.append(x)
        y_array.append(y)
        t += 1
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    return t_array, x_array, y_array
