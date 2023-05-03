
import os
import time
import numpy as np
import requests
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import datetime
from torchvision.transforms import ToTensor
import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Replace these with your own details
API_KEY = "f0ad32cc59f54ca21008c93d40fdad6d"
CITY = "Birmingham,GB"

# Function to fetch and display the weather information
def get_weather(root):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}"
    response = requests.get(url)
    weather_data = response.json()

    if weather_data["cod"] != "404":
        # Current weather
        current_weather = weather_data["list"][0]
        main = current_weather["main"]
        weather_desc = current_weather["weather"][0]["description"]
        temperature = main["temp"]
        temperature_celsius = temperature - 273.15
        
        weather_icon = current_weather["weather"][0]["icon"]
        icon_path = f"weather_icons/{weather_icon}@2x.png"
        weather_icon_img = Image.open(icon_path)
        weather_icon_photo = ImageTk.PhotoImage(weather_icon_img)
        weather_icon_label.config(image=weather_icon_photo)
        weather_icon_label.image = weather_icon_photo
        
        weather_label.config(text=f"Now: {weather_desc.capitalize()}\n       {temperature_celsius:.2f}°C")
        
        # 10-hour forecast with 2-hour increments
        forecast_frame = tk.Frame(root, bg="black")
        forecast_frame.pack(side="top", anchor="nw", padx=20, pady=30)

        title_label = tk.Label(forecast_frame, text="Next 10 hours weather forecast:", font=("Times New Roman", 30), bg="black", fg="white", anchor='w')
        title_label.grid(row=0, column=0, columnspan=5, sticky='w', padx=5, pady=0)

        # Add an empty row between the title and the icons
        empty_label = tk.Label(forecast_frame, text="", bg="black", fg="black")
        empty_label.grid(row=1, column=0)

        current_time = datetime.datetime.now()
        next_hours = [current_time + datetime.timedelta(hours=i*2) for i in range(1, 6)]

        for i, hour in enumerate(next_hours):
            index = i * 2 + 1  # Update the index to access 2-hour increments
            forecast = weather_data["list"][index]
            formatted_hour = hour.strftime("%H:%M")
            icon = forecast["weather"][0]["icon"]
            icon_path = f"weather_icons/{icon}@2x.png"
            icon_img = Image.open(icon_path)
            icon_photo = ImageTk.PhotoImage(icon_img)
            icon_label = tk.Label(forecast_frame, image=icon_photo, bg="black")
            icon_label.image = icon_photo
            icon_label.grid(row=2, column=i, padx=10, pady=0)

            hour_label = tk.Label(forecast_frame, text=formatted_hour, font=("Times New Roman", 20), bg="black", fg="white")
            hour_label.grid(row=3, column=i, padx=10, pady=0)

            temp = forecast["main"]["temp"]
            temp_celsius = temp - 273.15
            temp_label = tk.Label(forecast_frame, text=f"{temp_celsius:.1f}°C", font=("Times New Roman", 20), bg="black", fg="white")
            temp_label.grid(row=4, column=i, padx=10, pady=0)

    else:
        weather_label.config(text="Weather information not available")



# Function to show the GUI when a face is recognized
def create_GUI(root):
    root.title("Smart Mirror")
    root.configure(bg="black")
    root.attributes("-fullscreen", True)
    
    time_frame = tk.Frame(root, bg="black")
    time_frame.pack(side="top", anchor="nw", padx=20, pady=20)

    time_label = tk.Label(time_frame, text="", font=("Times New Roman", 45), bg="black", fg="white")
    time_label.grid(row=0, column=0, padx=30, pady=2)

    date_label = tk.Label(time_frame, text="", font=("Times New Roman", 45), bg="black", fg="white")
    date_label.grid(row=1, column=0, padx=30, pady=2)
    
    weather_frame = tk.Frame(root, bg="black")
    weather_frame.pack(side="top", anchor="nw", padx=20)

    global weather_icon_label
    weather_icon_label = tk.Label(weather_frame, bg="black")
    weather_icon_label.grid(row=0, column=1, padx=30, pady=10)

    global weather_label
    weather_label = tk.Label(weather_frame, text="", font=("Times New Roman", 45), bg="black", fg="white")
    weather_label.grid(row=0, column=0, padx=30, pady=10)

    def update_time():
        current_time = time.strftime("%H:%M:%S")
        time_label.config(text=current_time)
        root.after(1000, update_time)

    def update_date():
        current_date = datetime.date.today().strftime("%A, %B %d, %Y")
        date_label.config(text=current_date)
        root.after(60000, update_date)

    calendar_frame = tk.Frame(root, bg="black")
    calendar_frame.place(relx=1, rely=0, x=-20, y=20, anchor="ne")

    calendar_label = tk.Label(calendar_frame, text="", font=("Times New Roman", 30), bg="black", fg="white", wraplength=600, justify="right")
    calendar_label.pack(side="top", anchor="ne")

    def update_calendar():
        events = get_upcoming_events()
        event_text = "Reminders for Today:\n"
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            start_datetime = datetime.datetime.fromisoformat(start)
            start_time = start_datetime.strftime("%I:%M %p")  
            event_text += f" {event['summary']} ({start_time}) - \n"
        calendar_label.config(text=event_text)
        root.after(900000, update_calendar)  # Update calendar events every 15 minutes (900000 milliseconds)

    update_calendar()

    update_time()
    update_date()
    # Start the first update
    get_weather(root)
    create_quote_frame(root)
    root.mainloop()

def close_gui(root):
    root.destroy()

def get_random_quote():
    try:
        response = requests.get("https://zenquotes.io/api/random")
        quote_data = response.json()
        quote = quote_data[0]["q"]
        author = quote_data[0]["a"]
        return f"{quote}\n- {author}"
    except Exception as e:
        print(f"Error fetching quote: {e}")
        return "No quote available."

def create_quote_frame(root):
    quote_frame = tk.Frame(root, bg="black")
    quote_frame.place(relx=0, rely=1, x=20, y=-20, anchor="sw")

    quote_label = tk.Label(quote_frame, text="", font=("Times New Roman", 30), bg="black", fg="white", wraplength=1000, justify="left")
    quote_label.pack(side="bottom", anchor="se")

    def update_quote():
        quote = get_random_quote()
        quote_label.config(text=quote)
        root.after(3600000, update_quote)  # Update quote every hour (3600000 milliseconds)

    update_quote()


# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

def get_credentials():
    creds = None
    if os.path.exists('Smart_mirror_calender.json'):
        creds = Credentials.from_authorized_user_file('Smart_mirror_calender.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('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', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('Smart_mirror_calender.json', 'w') as token:
            token.write(creds.to_json())

    return creds

def get_upcoming_events():
    creds = get_credentials()
    service = build('calendar', 'v3', credentials=creds)

    now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    events_result = service.events().list(calendarId='primary', timeMin=now, maxResults=10, singleEvents=True, orderBy='startTime').execute()
    events = events_result.get('items', [])

    # Add this line to print the events
    print(f"Upcoming events: {events}")

    return events


# # initializing MTCNN and InceptionResnetV1 
mtcnn0 = MTCNN(image_size=100, margin=24, keep_all=False, min_face_size=100) # Empty MTCNN intialised: Keep all false means only 1 face will be detected
resnet = InceptionResnetV1(pretrained='vggface2').eval() #initializing the class and passing the pretrained model 'Vggface2' and will download if not already
dataset = datasets.ImageFolder('/Users/br/opt/anaconda3/envs/smart_mirror/The_photos') # reads the data from the folder and saves data in the data.pt file
index_to_class = {i:c for c,i in dataset.class_to_idx.items()} # returns the names of the folders that correspond to the images

def collate_fn(x):
    return x[0]

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t           
    avg = sum_num / len(num)
    return avg

def face_detection(loader):
    name_list = [] # list of names corresponding to cropped photos
    embedding_list = [] # list of embedding matrix after conversion from cropped faces to embedding matrix using resnet›››››››
    
    for image, index in loader:
        face, face_prob = mtcnn0(image, return_prob=True) #image is passed into the MTCNN's above and returns the face and probability
        if face is not None and face_prob>0.90:  #if the face is available and prob is > than 0.90 
            # Calculate embedding (unsqueeze to add batch dimension)
            emb = resnet(face.unsqueeze(0))  #then you pass the face into resnet(A CNN that can have thousand of layers and skip 1 or more layers for efficiency)
            embedding_list.append(emb.detach()) #gives the embedding (string of numbers that serves as a unique identifier)
            name_list.append(index_to_class[index])   # gives name list which will contain the names of all the folders
        else:
            print("No face detected")
            continue
    return name_list, embedding_list, face_prob

def minimum_distance(embedding_list, emb):
    dist_list = [] # list of matched distances, minimum distance
    for index, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    if len(dist_list) > 0:
        min_dist = min(dist_list) # get minumum dist value 
        min_index = dist_list.index(min_dist)
        name = name_list[min_index]
        return min_dist, name, dist_list, name_list
    else:
        return None, 'No face detected', None, []


def face_classification(frame):
    image_cropped, prob = mtcnn0(frame, return_prob=True) #image passed into MTCNN and returns face and probabilty
    if image_cropped is not None and prob > 0.60:
        emb = resnet(image_cropped.unsqueeze(0)).detach() 
        
        average_minimum_distance, name, min_dist, name_list = minimum_distance(embedding_list, emb)

        return average_minimum_distance, frame, name_list, name, min_dist
    

loader = DataLoader(dataset, collate_fn=collate_fn) #converts images into PIL image format for easier processing

name_list, embedding_list, face_prob = face_detection(loader) 

# # save data
data = [embedding_list, name_list] #combines the 2 lists into another list and will save save the data to
torch.save(data, 'data.pt') # data.pt file so that the compuation is not done repeatedly above

# # Using webcam recognize face
load_data = torch.load('data.pt') #Loading the data from data.pt
embedding_list = load_data[0] 
name_list = load_data[1] 

cam = cv2.VideoCapture(0) #when multiple webcams connected, you use the first one

print("starting!!")

def show_gui():
    index = 0

    while index < 100:
        ret, frame = cam.read()

        if not ret:
            print("fail to grab frame, try again")
            break

        frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)

        result = face_classification(frame)
        if result is not None:
            avg_min_dist, frame, name_list, name, min_dist = result
            min_distance = min(min_dist)  # Calculate the minimum distance
            if min_distance <= 0.7:  # Customize this threshold value according to your requirements
                print("Face matched with:", name)
                print("The Minimum distance to your face is: ", min_distance)
                break
            else:
                print("No matching face found.")
                
        else:
            print("No face detected")

        time.sleep(1)  # Try to recognize a face every 1 second
    
    # Call the GUI function to display the GUI
    root = tk.Tk()
    create_GUI(root)
    
    # Schedule GUI to close after 1 minute (60000 milliseconds)
    root.after(60000, close_gui, root)
    root.mainloop()

print("starting!!")

# Call the show_gui function to start the face recognition and display the GUI
show_gui()

cam.release()
cv2.destroyAllWindows()