import cv2  # opencv-contrib-python
import numpy as np
import matplotlib.pyplot as plt
import methods
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

point = None
bbox = None
is_selected = False
select_method_text = "Select a method for tracking"


def select_point(event, x, y, flags, param):
    global point, is_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        point = np.array([[x, y]], dtype=np.float32)
        is_selected = True


def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        video_file_var.set(file_path)


def save_file(var, file_extension, default_name):
    file_path = filedialog.asksaveasfilename(defaultextension=file_extension,
                                             filetypes=[(file_extension, f"*{file_extension}")],
                                             initialfile=default_name)
    if file_path:
        var.set(file_path)


def main():
    global point, is_selected, bbox

    METHOD = dropdown_var.get()
    VIDEO_NAME = video_file_var.get()
    OUTPUT_NAME = output_video_var.get()
    GRAPH_NAME = output_graph_var.get()

    if not METHOD or METHOD == select_method_text or not VIDEO_NAME or not OUTPUT_NAME or not GRAPH_NAME:
        messagebox.showerror("Input Error", "All fields are required")
        return
    root.destroy()  # Close the window

    cap = cv2.VideoCapture(VIDEO_NAME)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(OUTPUT_NAME, cv2.VideoWriter.fourcc(*'mp4v'), fps,
                          (frame_width, frame_height))

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        cap.release()
        exit()

    if METHOD in methods.optical_flow_methods:
        cv2.namedWindow("Set Tracking Point", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Set Tracking Point", select_point)
        while not is_selected:
            cv2.imshow("Set Tracking Point", first_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyWindow("Set Tracking Point")

    elif METHOD in methods.trackers:
        cv2.namedWindow("Set Bounding Box", cv2.WINDOW_NORMAL)
        while not is_selected:
            cv2.putText(first_frame,
                        'Draw a bounding box and then press SPACE or ENTER button! Cancel by pressing C or ESC button!',
                        (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            bbox = cv2.selectROI("Set Bounding Box", first_frame, fromCenter=False, showCrosshair=True)
            x, y, w, h = bbox
            point = (x + w // 2, y + h // 2, w, h)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            is_selected = True
        cv2.destroyWindow("Set Bounding Box")

    t_array = []
    x_array = []
    y_array = []

    if METHOD in methods.trackers:
        t_array, x_array, y_array = methods.general_tracker(first_frame, bbox, cap, out, METHOD)

    elif METHOD in methods.optical_flow_methods:
        t_array, x_array, y_array = methods.optical_flow_tracker(first_frame, point, cap, out, METHOD)

    plt.figure(figsize=(15, 15))
    plt.plot(t_array, y_array, label='Y Axis', linestyle='-', color='y')
    plt.plot(t_array, x_array, label='X Axis', linestyle='-', color='r')
    plt.title(METHOD)
    plt.xlabel("Time")
    plt.ylabel("Coordinates")
    plt.grid(True)
    plt.legend()
    plt.savefig(GRAPH_NAME)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


root = tk.Tk()
root.title("Tracking Toolbox")

dropdown_var = tk.StringVar()
options = np.append(methods.trackers, methods.optical_flow_methods)
dropdown_menu = ttk.Combobox(root, textvariable=dropdown_var, values=list(options), state="readonly", width=50)
dropdown_menu.grid(row=0, column=1, padx=10, pady=10)
dropdown_menu.set(select_method_text)

video_file_var = tk.StringVar()
video_file_button = ttk.Button(root, text="Select Video File", command=select_file)
video_file_button.grid(row=1, column=0, padx=10, pady=10)
video_file_label = ttk.Entry(root, textvariable=video_file_var, state="readonly", width=50)
video_file_label.grid(row=1, column=1, padx=10, pady=10)

output_video_var = tk.StringVar()
output_video_button = ttk.Button(root, text="Save Output Video",
                                 command=lambda: save_file(output_video_var, ".mp4", "output_video"))
output_video_button.grid(row=2, column=0, padx=10, pady=10)
output_video_label = ttk.Entry(root, textvariable=output_video_var, state="readonly", width=50)
output_video_label.grid(row=2, column=1, padx=10, pady=10)

output_graph_var = tk.StringVar()
output_graph_button = ttk.Button(root, text="Save Output Graph",
                                 command=lambda: save_file(output_graph_var, ".png", "output_graph"))
output_graph_button.grid(row=3, column=0, padx=10, pady=10)
output_graph_label = ttk.Entry(root, textvariable=output_graph_var, state="readonly", width=50)
output_graph_label.grid(row=3, column=1, padx=10, pady=10)

start_button = ttk.Button(root, text="Start Tracking", command=main)
start_button.grid(row=4, column=1, padx=10, pady=10)

root.mainloop()
