# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 03:39:03 2023
FaceSwap v.1.4
Updated on: 31.01.2023
original method based on paper "The contribution of different face parts to deep face recognition"[1]

@author: N.H. Lestriandoko

Update: 
    1. new GUI
    2. Update "UNDO" function
    3. Help/hint features
    4. advance landmark points 

[1] Lestriandoko, N. H., Veldhuis, R., & Spreeuwers, L. (2022). 
The contribution of different face parts to deep face recognition. 
Frontiers in Computer Science. https://doi.org/10.3389/fcomp.2022.958629    

"""

import PySimpleGUI as sg
import os.path
#import time

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import dlib
import io
import face_swap as fs
import folder_swap as fds
import face_detection as fd
import average_face as af
import tempfile
from PIL import Image

img1 =[]
img2 =[]
tmp_file1 = tempfile.NamedTemporaryFile(suffix=".png").name
tmp_file2 = tempfile.NamedTemporaryFile(suffix=".png").name

matplotlib.use("TkAgg")

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def draw_image(image_file):
    if os.path.exists(image_file):
        image = Image.open(image_file)
        image.thumbnail((300, 300))
        #image.thumbnail((500, 500))
        return image

def open_folder_swap_window(out_folder="./results/", dlib_folder = './dlib-models-master/',folder_type=1):
    file_list_swap = [
        [
            sg.Text("Input Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER INPUT SWAP-"),
            sg.FolderBrowse(size=(10,1)),
        ],
        [
            sg.Listbox(values=[], enable_events=True, size=(40, 10), key="-FILELIST INPUT-")
        ],
        [sg.HSeparator()],
        [sg.Text(size=(1,20)), sg.Image(key="-VIEW INPUT IMAGE-")],
    ]
    results_list_swap = [
        [
            sg.Text("Result Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER RESULT-"),
            sg.FolderBrowse(size=(10,1)),
        ],
        [
            sg.Listbox(values=[], enable_events=True, size=(40, 10), key="-FILELIST RESULT-")
        ],
        [sg.HSeparator()],
        [sg.Text(size=(1,20)), sg.Image(key="-VIEW RESULT IMAGE-")],
    ]
    avg_list_view = [
        [
            sg.Text("Average Face"),
            sg.In(size=(25, 1), enable_events=True, key="-FILE AVG-"),
            sg.FileBrowse(size=(10, 1), file_types=(("PNG files", "*.png"),("JPG files", "*.jpg")))
        ],
        [
            sg.Radio("Full Face", "RadioSwap", size=(10, 1), key="-RDO FULLFACE-", enable_events=True, default=False),
            sg.Combo(
                ["with beard", "without beard", "without beard and cheek"],
                default_value="With beard",
                key="-COMBO FULLFACE-",
                enable_events=True,
                readonly=True,
                disabled=True,
            ),
            
        ],
        [
            sg.Radio("Face Parts", "RadioSwap", size=(10, 1), key="-RDO FACEPARTS-", enable_events=True, default=True),
            #sg.Checkbox('Full face   OR ', key="-FULL FACE-", enable_events=True, default=False),
            sg.Text("Face Parts",key="-TEXT PARTS-"),
            sg.Combo(
                ["eyebrows", "eyes","nose","mouth","eyebrows-eyes","eyebrows-nose","eyebrows-mouth","eyes-nose","eyes-mouth","nose-mouth","eyebrows-eyes-nose","eyebrows-eyes-mouth","eyebrows-nose-mouth","eyes-nose-mouth","eyebrows-eyes-nose-mouth"],
                default_value="eyebrows-eyes-nose-mouth",
                key="-COMBO PARTS-",
                enable_events=True,
                readonly=True,
            ),
        ],
        [
             sg.Text("     Method"),
             sg.Combo(["NDTS", "NDT", "NDS"],
                      default_value = "NDTS",
                      key = "-COMBO METHOD-",
                      enable_events=True,
                      readonly=True,
                      ),
            sg.Text("Replace Texture and Shape", size=(32, 1), key="-TEXT_METHOD-"),
        ],     
        [sg.HSeparator()],
        [sg.Text(size=(40,1))],
        [sg.Text(size=(1,25)),sg.Image(key="-VIEW AVG FILE-")],
        [   
            sg.Button("Replace Average Face with Dataset", key="-BTN SWAP AVG-"),
            sg.Button("Replace Dataset with Average Face", key="-BTN SWAP DATASET-")
        ],
    ]
    layout = [
        [sg.Column(avg_list_view),
         sg.VSeperator(),
         sg.Column(file_list_swap),
         sg.VSeperator(),
         sg.Column(results_list_swap),
        ],
    ]
    window = sg.Window("Average Face Replacement", layout, modal=True)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "-FILE AVG-":
            filename = values["-FILE AVG-"]
            image = draw_image(filename)
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-VIEW AVG FILE-"].update(data=bio.getvalue())
        
        elif event == "-FOLDER INPUT SWAP-":
            folder = values["-FOLDER INPUT SWAP-"]
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []
        
            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".png", ".jpg"))
            ]
            window["-FILELIST INPUT-"].update(fnames)
        
        elif event == "-FILELIST INPUT-":    
            try:
                filename = os.path.join(
                    values["-FOLDER INPUT SWAP-"], values["-FILELIST INPUT-"][0]
                )
                image = draw_image(filename)
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-VIEW INPUT IMAGE-"].update(data=bio.getvalue())
            except:
                pass

        
        elif event == "-COMBO METHOD-":
            swap_method = values["-COMBO METHOD-"]
            if swap_method == "NDTS":
                window["-TEXT_METHOD-"].update("Replace Texture and Shape")
            elif swap_method == "NDT":
                window["-TEXT_METHOD-"].update("Replace Texture Only")
            elif swap_method == "NDS":
                window["-TEXT_METHOD-"].update("Replace Shape Only") 
        
        elif event == "-BTN SWAP AVG-":
            #sg.popup("-BTN SWAP AVG-")
            input_folder = values["-FOLDER INPUT SWAP-"]
            output_folder = values["-FOLDER RESULT-"]
            input_avgFace = values["-FILE AVG-"]
            parts = values["-COMBO PARTS-"]
            if values["-RDO FULLFACE-"] and values["-COMBO FULLFACE-"]=="with beard":
                sg.popup("-FULL FACE-")
                method = "full-face"
            elif values["-RDO FULLFACE-"] and values["-COMBO FULLFACE-"]=="without beard": 
                sg.popup("-HALF FACE CHEEK-")
                method = "half-face-cheek-v2"
            elif values["-RDO FULLFACE-"] and values["-COMBO FULLFACE-"]=="without beard and cheek": 
                sg.popup("-HALF FACE-")
                method = "half-face"
            else:
                method = values["-COMBO METHOD-"]
            bInverse = True
            fd_swap = fds.folder_swap(input_folder, output_folder, input_avgFace, folder_type=folder_type, parts=parts, method=method, bInverse=bInverse, dlib_path=dlib_folder)
            fd_swap.swap()

        elif event == "-BTN SWAP DATASET-":
            #sg.popup("-BTN SWAP DATASET-")
            input_folder = values["-FOLDER INPUT SWAP-"]
            output_folder = values["-FOLDER RESULT-"]
            input_avgFace = values["-FILE AVG-"]
            parts = values["-COMBO PARTS-"]
            if values["-RDO FULLFACE-"] and values["-COMBO FULLFACE-"]=="with beard": 
                method = "full-face"
            elif values["-RDO FULLFACE-"] and values["-COMBO FULLFACE-"]=="without beard": 
                sg.popup("-HALF FACE CHEEK-")
                method = "half-face-cheek-v2"
            elif values["-RDO FULLFACE-"] and values["-COMBO FULLFACE-"]=="without beard and cheek": 
                sg.popup("-HALF FACE-")
                method = "half-face"
            else:
                method = values["-COMBO METHOD-"]
            bInverse = False
            fd_swap = fds.folder_swap(input_folder, output_folder, input_avgFace, folder_type=folder_type, parts=parts, method=method, bInverse=bInverse, dlib_path=dlib_folder)
            fd_swap.swap()

        elif values["-RDO FULLFACE-"]:
            #sg.popup("-RDO FULLFACE-")
            window["-COMBO FULLFACE-"].update(disabled=False)
            window["-COMBO PARTS-"].update(disabled=True)
            window["-COMBO METHOD-"].update(disabled=True, value = "NDTS")
            window["-TEXT_METHOD-"].update("Replace Texture and Shape")

        elif values["-RDO FACEPARTS-"]:
            #sg.popup("-RDO FACEPARTS-")
            window["-COMBO PARTS-"].update(disabled=False)
            window["-COMBO METHOD-"].update(disabled=False)
            window["-COMBO FULLFACE-"].update(disabled=True)
            
    window.close()

def open_avg_window(out_folder="./results/", dlib_folder = './dlib-models-master/'):
    file_list_avg = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER AVG-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST AVG-"
        )
    ],
    ]
    
    image_viewer_avg = [
        [sg.Text("Output folder: "+ out_folder )],
        [sg.Text(size=(40, 1), key="-TOUT AVG FILE-")],
        [sg.Image(key="-VIEW IMAGE-")],
        [sg.HSeparator()],
        [sg.Button("Calculate Average Face", key="-CALCULATE AVG-")],
        #[sg.Image(key="-VIEW AVG IMAGE-")],
    ]
    
    layout = [
        [
            sg.Checkbox('Face Detection', key="-DETECT FACE-", enable_events=True, default=True),
            sg.Slider(range=(0.25,0.9), default_value=0.6, resolution=.05, size=(30,15), orientation='horizontal', key="-PADDING SLIDER-"),
            sg.Text("Cropping scale"),
        ],
        [
            sg.Text("Image Resolution", key="-AVG Res-"),
            sg.Radio("150x150", "Radio", size=(10, 1), key="-AVG 150-", default=True),
            sg.Radio("320x320", "Radio", size=(10, 1), key="-AVG 320-", default=False),
            sg.Radio("500x500", "Radio", size=(10, 1), key="-AVG 500-", default=False),                        
        ],
        [
            sg.Text("Folder type", key="-FOLDER TYPE TXT-"),
            sg.Combo(["1 Level Folder", "2 Level Folder"],
                      default_value = "1 Level Folder",
                      key = "-FOLDER TYPE-",
                      enable_events=True,
                      readonly=True,
                      ),
        ],
        [sg.Column(file_list_avg),
         sg.VSeperator(),
         sg.Column(image_viewer_avg),
         sg.VSeperator(),
         sg.Image(key="-VIEW AVG IMAGE-")
        ],
        [
            sg.HSeparator(),
        ],
        [sg.ProgressBar(10, orientation='h', size=(61, 10), key='progressbar'),sg.Text(f'{0}%', size=(6,1), key='-%-')],
        
    ]
    window = sg.Window("Average Face Calculation", layout, modal=True)
    progress_bar = window['progressbar']
    
    key_list = "-AVG 150-", "-AVG 320-", "-AVG 500-"
    #for key in key_list:
    #    window[key].update(disabled=True)
    
    #choice = None
    while True:
        event, values = window.read()
        progress_bar.UpdateBar(0)        
                
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        elif event == "-DETECT FACE-":
            if values["-DETECT FACE-"] == True:            
                #sg.popup("values[-DETECT FACE-]=True")
                for key in key_list:
                    window[key].update(disabled=False)
            
            elif values["-DETECT FACE-"] == False:
                #sg.popup("values[-DETECT FACE-]=False")
                for key in key_list:
                    window[key].update(disabled=True)
        
        
        elif event == "-FOLDER AVG-":
            folder_avg = values["-FOLDER AVG-"]
            #sg.popup("event = "+folder_avg)
            #print("event = "+folder_avg)
            try:
                # Get list of files in folder
                file_list = os.listdir(folder_avg)
            except:
                file_list = []
        
            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder_avg, f))
                and f.lower().endswith((".png", ".jpg"))
            ]
            window["-FILE LIST AVG-"].update(fnames)
        
        elif event == "-FILE LIST AVG-":  # A file was chosen from the listbox
            try:
                filename = os.path.join(
                    values["-FOLDER AVG-"], values["-FILE LIST AVG-"][0]
                )
                window["-TOUT AVG FILE-"].update(filename)
                image = draw_image(filename)
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-VIEW IMAGE-"].update(data=bio.getvalue())
                #window["-VIEW IMAGE-"].update(filename=filename)
            except:
                pass
    
        elif event == "-CALCULATE AVG-":
            if values["-FOLDER TYPE-"] == "1 Level Folder":
                folder_type=1
            else:
                folder_type=2
            
            if not os.path.exists(values["-FOLDER AVG-"]):
                sg.popup('Warning',"Image Folder is not exist!")
            elif values["-FOLDER AVG-"]=="" :
                sg.popup('Warning',"Image folder is empty!")
            else:
                if values["-DETECT FACE-"] :
                
                    #sg.popup("Detecting face(s)..")
                    window["-%-"].update(f'{10}%')
                    progress_bar.UpdateBar(1,10)
                    padding = values["-PADDING SLIDER-"]
                    aligned_directory = os.path.join(out_folder, r'Aligned')
                    if not os.path.exists(aligned_directory):
                        os.makedirs(aligned_directory)
                    if values["-AVG 150-"] == True:
                        detect_face = fd.face_detection(values["-FOLDER AVG-"], aligned_directory, folder_type=folder_type, padding=padding, size=150, dlib_path = dlib_folder+'/')
                        detect_face.detect_faces()
                    elif values["-AVG 320-"] == True:
                        detect_face = fd.face_detection(values["-FOLDER AVG-"], aligned_directory, folder_type=folder_type, padding=padding, size=320, dlib_path = dlib_folder+'/')
                        detect_face.detect_faces()
                    elif values["-AVG 500-"] == True:
                        detect_face = fd.face_detection(values["-FOLDER AVG-"], aligned_directory, folder_type=folder_type, padding=padding, size=500, dlib_path = dlib_folder+'/')
                        detect_face.detect_faces()
                    
                    window["-%-"].update(f'{50}%')
                    progress_bar.UpdateBar(5,10)
                    #sg.popup("Calculating the average of facial landmarks..")
                    # average facial landmarks
                    meanPoints_directory = os.path.join(out_folder, r'AverageFacialLandmarks')
                    if not os.path.exists(meanPoints_directory):
                        os.makedirs(meanPoints_directory)
                    avg_faces = af.average_face(aligned_directory, meanPoints_directory +'/meanPoints.npz',folder_type=folder_type, dlib_path = dlib_folder+'/')
                    avg_faces.average_points()
                    window["-%-"].update(f'{60}%')
                    progress_bar.UpdateBar(6,10)

                    # Facial landmarks morphing
                    
                    #sg.popup("Realigning images..")
                    realign_directory = os.path.join(out_folder, r'Realign')
                    if not os.path.exists(realign_directory):
                        os.makedirs(realign_directory)
                    avg_faces = af.average_face(aligned_directory, realign_directory, folder_type=folder_type, mean_points = meanPoints_directory +'/meanPoints.npz', dlib_path = dlib_folder+'/')
                    avg_faces.manual_align()
                    window["-%-"].update(f'{80}%')
                    progress_bar.UpdateBar(8,10)

                else:
                    #sg.popup("Calculating the average of facial landmarks..")
                    window["-%-"].update(f'{10}%')
                    progress_bar.UpdateBar(1)                    
                    # average facial landmarks
                    meanPoints_directory = os.path.join(out_folder, r'AverageFacialLandmarks')
                    if not os.path.exists(meanPoints_directory):
                        os.makedirs(meanPoints_directory)
                    avg_faces = af.average_face(values["-FOLDER AVG-"], meanPoints_directory +'/meanPoints.npz',folder_type=folder_type, dlib_path = dlib_folder+'/')
                    avg_faces.average_points()
                    window["-%-"].update(f'{30}%')
                    progress_bar.UpdateBar(3,10)
                
                    # Facial landmarks morphing
                    
                    #sg.popup("Realigning images..")
                    realign_directory = os.path.join(out_folder, r'Realign')
                    if not os.path.exists(realign_directory):
                        os.makedirs(realign_directory)
                    avg_faces = af.average_face(values["-FOLDER AVG-"], realign_directory, folder_type=folder_type, mean_points = meanPoints_directory +'/meanPoints.npz', dlib_path = dlib_folder+'/')
                    avg_faces.manual_align()
                    window["-%-"].update(f'{70}%')
                    progress_bar.UpdateBar(7,10)
                
                # Calculate average face
                #sg.popup("Averaging faces..")
                final_directory = os.path.join(out_folder, r'AverageFace')
                if not os.path.exists(final_directory):
                    os.makedirs(final_directory)
                progress_bar.UpdateBar(8,10)
                avg_faces = af.average_face(realign_directory, final_directory +'/meanFace.npz',folder_type=folder_type, dlib_path = dlib_folder+'/')
                avg_faces.average_faces()
                progress_bar.UpdateBar(9,10)
                filename = final_directory +'/meanFace.png'
                window["-VIEW AVG IMAGE-"].update(filename=filename)
                window["-%-"].update(f'{100}%')
                progress_bar.UpdateBar(10,10)
    window.close()
    
def main():
    # First the window layout in 2 columns
    file_list_column = [
        [
            sg.Text("Image1 Path"),
            sg.In(size=(50, 1), enable_events=True, key="-FILE1-"),
            sg.FileBrowse(size=(10, 1), file_types=(("PNG files", "*.png"),("JPG files", "*.jpg"))),
        ],
        [sg.Text(size=(60, 1), key="-TOUT1-")],
        [sg.Text(size=(1,20)),sg.Image(key="-IMAGE1-")],
        [sg.Button("Detect Face 1", key="-DetectFace 1-",disabled=True),sg.Button("Swap 1", key="-SWAP 1-", disabled=True),sg.Button("Undo",key="-UNDO 1-",disabled=True)],
        
        #[sg.Canvas(key="-Result1-")],
        [sg.Image(key="-Result1-")],
    ]
    
    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [
            sg.Text("Image2 Path"),
            sg.In(size=(50, 1), enable_events=True, key="-FILE2-"),
            sg.FileBrowse(size=(10, 1), file_types=(("PNG files", "*.png"),("JPG files", "*.jpg"))),
        ],
        [sg.Text(size=(60, 1), key="-TOUT2-")],
        [sg.Text(size=(1,20)),sg.Image(key="-IMAGE2-")],
        [sg.Button("Detect Face 2", key="-DetectFace 2-",disabled=True),sg.Button("Swap 2", key="-SWAP 2-", disabled=True),sg.Button("Undo",key="-UNDO 2-",disabled=True)],
        
        [sg.Image(key="-Result2-")],
    ]
    
    # ----- Full layout -----
    layout = [
        [
            sg.Text("Face Parts"),
            sg.Combo(
                ["eyebrows", "eyes","nose","mouth","eyebrows-eyes","eyebrows-nose","eyebrows-mouth","eyes-nose","eyes-mouth","nose-mouth","eyebrows-eyes-nose","eyebrows-eyes-mouth","eyebrows-nose-mouth","eyes-nose-mouth","eyebrows-eyes-nose-mouth"],
                default_value="eyebrows-eyes-nose-mouth",
                key="-PARTS-",
                enable_events=True,
                readonly=True,
            ),
            sg.Text("DLIB model path"),
            sg.In('./dlib-models-master/', size=(25, 1), enable_events=True, key="-DLIBpath-"),
            sg.FolderBrowse(),
            sg.Text(size=(40, 1), key="-TOUT-"),
            sg.Button("Calculate Average Face", key="-OPEN AVG-")
            #sg.Input(begin_y, size=(5, 1), key=key2, enable_events=True),
        ],    
        [
             sg.Text("Method"),
             sg.Combo(["NDTS", "NDT", "NDS"],
                      default_value = "NDTS",
                      key = "-METHOD-",
                      enable_events=True,
                      readonly=True,
                      ),
            sg.Text("Replace Texture and Shape", size=(32, 1), key="-TOUT_METHOD-"),
            sg.Text("Save Output to"),
            sg.In('./results/', size=(40, 1), enable_events=True, key="-OUTPUTpath-"),
            sg.FolderBrowse(),
            sg.Text(size=(50, 1), key="-TOUTPUTPATH-")
        ],
        [
            sg.Text("Image Resolution"),
            sg.Radio("150x150", "Radio", size=(10, 1), key="-150-", default=True),
            sg.Radio("320x320", "Radio", size=(10, 1), key="-320-", default=False),
            sg.Radio("500x500", "Radio", size=(10, 1), key="-500-", default=False),
            sg.Text("Folder type", key="-MAIN FOLDER TYPE TXT-"),
            sg.Combo(["Non Folder", "1 Level Folder", "2 Level Folder"],
                      default_value = "Non Folder",
                      key = "-MAIN FOLDER TYPE-",
                      enable_events=True,
                      readonly=True,
                      ),
            sg.Button("Replace Images with Average Face", key="-OPEN SWAP-", disabled = True)                        
        ],
        [
            sg.HSeparator(),
        ],
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]
    
    window = sg.Window("Vis-a-vis desktop v.1.6", layout)
    undo1 = ""
    undo2 = ""
    state1 = 0
    state2 = 0
    # Run the Event Loop
    while True:
        event, values = window.read()
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        if event == "-FILE1-":  
            try:
                filename = values["-FILE1-"]
                window["-TOUT1-"].update(filename)
                image = draw_image(filename)
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE1-"].update(data=bio.getvalue())
                #window["-IMAGE1-"].update(filename=filename)
                window["-SWAP 1-"].update(disabled=True)
                window["-DetectFace 1-"].update(disabled=False)
                undo1 = filename
                window["-UNDO 1-"].update(disabled=True)
                state1 = 0 # open an image
                
            except:
                pass
    
        elif event == "-FILE2-":  
            try:
                filename = values["-FILE2-"]
                window["-TOUT2-"].update(filename)
                image = draw_image(filename)
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE2-"].update(data=bio.getvalue())
                #window["-IMAGE2-"].update(filename=filename)
                window["-SWAP 2-"].update(disabled=True)
                window["-DetectFace 2-"].update(disabled=False)
                undo2 = filename
                window["-UNDO 2-"].update(disabled=True)
                state2 = 0 # open an image
            except:
                pass
        
        elif event == "-OPEN AVG-":
            open_avg_window(out_folder = values["-OUTPUTpath-"], dlib_folder = values["-DLIBpath-"])

        elif event == "-OPEN SWAP-":
            if values["-MAIN FOLDER TYPE-"]== "1 Level Folder":
                folder_type = 1
            else:
                folder_type = 2
            
            open_folder_swap_window(out_folder = values["-OUTPUTpath-"], dlib_folder = values["-DLIBpath-"], folder_type = folder_type)
                
        elif event == "-DLIBpath-":
            folder = values["-DLIBpath-"]
            window["-TOUT-"].update(folder)

        elif event == "-METHOD-":
            swap_method = values["-METHOD-"]
            if swap_method == "NDTS":
                window["-TOUT_METHOD-"].update("Replace Texture and Shape")
            elif swap_method == "NDT":
                window["-TOUT_METHOD-"].update("Replace Texture Only")
            elif swap_method == "NDS":
                window["-TOUT_METHOD-"].update("Replace Shape Only") 
        
        elif event == "-SWAP 1-":  
            
            filename1 = values["-FILE1-"]
            filename2 = values["-FILE2-"]
            dlib_folder = values["-DLIBpath-"]
            method = values["-METHOD-"]
            out_folder = values["-OUTPUTpath-"]
            
            swap_face = fs.face_swap(filename1, filename2, parts = values["-PARTS-"], method = method, dlib_path = dlib_folder+'/')
            result1 = swap_face.swap()
            out_filename = out_folder +'/result1.png'
            dlib.save_image(result1, out_filename)

            window["-FILE1-"].update(out_filename)
            window["-TOUT1-"].update(out_filename)
            window["-IMAGE1-"].update(filename=out_filename)
            #window["-Result1-"].update(filename=out_filename)
            window["-UNDO 1-"].update(disabled=False)
            window["-DetectFace 1-"].update(disabled=True)
            undo1 = filename1
            state1 = 2 # replacement
            #fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
            #fig.add_subplot(111).imshow(result1)
            #draw_figure(window["-Result1-"].TKCanvas, fig)
    
        elif event == "-SWAP 2-":  
            filename1 = values["-FILE1-"]
            filename2 = values["-FILE2-"]
            dlib_folder = values["-DLIBpath-"]
            method = values["-METHOD-"]
            out_folder = values["-OUTPUTpath-"]

            swap_face = fs.face_swap(filename2, filename1, parts = values["-PARTS-"], method = method, dlib_path = dlib_folder+'/')
            result2 = swap_face.swap()
            out_filename = out_folder +'/result2.png'
            dlib.save_image(result2, out_filename)

            window["-FILE2-"].update(out_filename)
            window["-TOUT2-"].update(out_filename)
            window["-IMAGE2-"].update(filename=out_filename)
            #window["-Result2-"].update(filename=out_filename)
            window["-UNDO 2-"].update(disabled=False)
            window["-DetectFace 2-"].update(disabled=True)
            undo2 = filename2
            state2 = 2 # replacement

        elif event == "-DetectFace 1-":
            window["-SWAP 1-"].update(disabled=False)
            filename = values["-FILE1-"]
            
            dlib_folder = values["-DLIBpath-"]
            out_folder = values["-OUTPUTpath-"]
            out_filename = out_folder +'/face_1.png'
            if values["-150-"] == True:
                detect_face = fd.face_detection(filename, out_filename, folder_type=0, padding=0.60, size=150, dlib_path = dlib_folder+'/')
                detect_face.detect_faces()
            elif values["-320-"] == True:
                detect_face = fd.face_detection(filename, out_filename, folder_type=0, padding=0.80, size=320, dlib_path = dlib_folder+'/')
                detect_face.detect_faces()
            elif values["-500-"] == True:
                detect_face = fd.face_detection(filename, out_filename, folder_type=0, padding=0.60, size=500, dlib_path = dlib_folder+'/')
                detect_face.detect_faces()

            window["-FILE1-"].update(out_filename)
            window["-TOUT1-"].update(out_filename)
            
            image = draw_image(out_filename)
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-IMAGE1-"].update(data=bio.getvalue())
            undo1 = filename
            window["-UNDO 1-"].update(disabled=False)
            window["-DetectFace 1-"].update(disabled=True)
            state1 = 1 # detect a face
                
        elif event == "-DetectFace 2-":
            window["-SWAP 2-"].update(disabled=False)
            filename = values["-FILE2-"]
            
            dlib_folder = values["-DLIBpath-"]
            out_folder = values["-OUTPUTpath-"]
            out_filename = out_folder +'/face_2.png'
            if values["-150-"] == True:
                detect_face = fd.face_detection(filename, out_filename, folder_type=0, padding=0.60, size=150, dlib_path = dlib_folder+'/')
                detect_face.detect_faces()
            elif values["-320-"] == True:
                detect_face = fd.face_detection(filename, out_filename, folder_type=0, padding=0.60, size=320, dlib_path = dlib_folder+'/')
                detect_face.detect_faces()
            elif values["-500-"] == True:
                detect_face = fd.face_detection(filename, out_filename, folder_type=0, padding=0.60, size=500, dlib_path = dlib_folder+'/')
                detect_face.detect_faces()

            window["-FILE2-"].update(out_filename)
            window["-TOUT2-"].update(out_filename)

            image = draw_image(out_filename)
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-IMAGE2-"].update(data=bio.getvalue())
            undo2 = filename
            window["-UNDO 2-"].update(disabled=False)
            window["-DetectFace 2-"].update(disabled=True)
            state2 = 1 # detect a face
            
        elif event == "-UNDO 1-":
            try:
                filename = undo1
                window["-TOUT1-"].update(filename)
                window["-FILE1-"].update(filename)

                image = draw_image(filename)
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE1-"].update(data=bio.getvalue())

                #window["-IMAGE1-"].update(filename=filename)
                if state1==1: 
                    window["-SWAP 1-"].update(disabled=True)
                    window["-DetectFace 1-"].update(disabled=False)
                    window["-UNDO 1-"].update(disabled=True)
                if state1==2: 
                    window["-SWAP 1-"].update(disabled=False)
                    window["-DetectFace 1-"].update(disabled=True)
                    window["-UNDO 1-"].update(disabled=True)
            except:
                pass
            
        elif event == "-UNDO 2-":
            try:
                filename = undo2
                window["-TOUT2-"].update(filename)
                window["-FILE2-"].update(filename)

                image = draw_image(filename)
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE2-"].update(data=bio.getvalue())

                #window["-IMAGE2-"].update(filename=filename)
                if state2==1: 
                    window["-SWAP 2-"].update(disabled=True)
                    window["-DetectFace 2-"].update(disabled=False)
                    window["-UNDO 2-"].update(disabled=True)
                if state2==2: 
                    window["-SWAP 2-"].update(disabled=False)
                    window["-DetectFace 2-"].update(disabled=True)
                    window["-UNDO 2-"].update(disabled=True)
            except:
                pass
            
        elif event == "-MAIN FOLDER TYPE-":
            if values["-MAIN FOLDER TYPE-"] == "Non Folder":
                window["-OPEN SWAP-"].update(disabled=True)
            else:
                window["-OPEN SWAP-"].update(disabled=False)
                
    window.close()
    
if __name__ == "__main__":
    main()
    
