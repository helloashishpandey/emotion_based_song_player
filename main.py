import tkinter as tk
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pyaudio
import wave
import tensorflow as tf
from pydub import AudioSegment  
from matplotlib.animation import FuncAnimation

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model = tf.keras.models.load_model('path/model/model.h5')

# Song folders based on emotion
song_dirs = {
    'Angry': 'path/songs/Angry',
    'Fearful': 'path/songs/Fearful',
    'Happy': 'path/songs/Happy',
    'Sad': 'path/songs/Sad',
    'Surprised': 'path/songs/Surprised',
    'Disgusted': 'path/songs/Disgusted',
    'Neutral': 'path/songs/Neutral'
}


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (48, 48))
    processed_image = np.expand_dims(resized_image, axis=-1)
    processed_image = processed_image.astype('float32') / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

def play_song(file_path):
    if not os.path.exists(file_path):
        print(f"Error finding or playing song: {file_path}")
        return

    try:
        os.startfile(file_path)
    except FileNotFoundError as e:
        print(f"Error playing song: {e}")

def capture_and_predict():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image.")
        return

    cv2.imshow("Captured Image", frame)

    processed_image = preprocess_image(frame)
    prediction = model.predict(processed_image)
    emotion_index = np.argmax(prediction)
    emotions = list(song_dirs.keys())
    predicted_emotion = emotions[emotion_index]

    print(f"Predicted Emotion: {predicted_emotion}")
    play_song_randomly(predicted_emotion)
    update_plot(frame)

def play_song_randomly(emotion):
    try:
        songs = os.listdir(song_dirs[emotion])
        print(f"Songs available for {emotion}: {songs}")
        if not songs:
            print(f"No songs available for emotion: {emotion}")
            return
        
        selected_song = random.choice(songs)
        print(f"Selected song: {selected_song}")
        mp3_path = song_dirs[emotion].replace("\\", "/") + '/' + selected_song
        print(f"MP3 path: {mp3_path}")
    
        play_song(mp3_path)
    
    except (FileNotFoundError, wave.Error) as e:
        print(f"Error finding or playing song: {e}")


FORMAT = pyaudio.paInt16  
CHANNELS = 1             
RATE = 44100            
CHUNK = 1024             


p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)



def update_plot(frame):
    data = stream.read(CHUNK)
    
    audio_data = np.frombuffer(data, dtype=np.int16)
    fft_data = np.fft.fft(audio_data)
    fft_data = np.abs(fft_data)[:CHUNK // 2] 
    
    line.set_ydata(fft_data)
    
    dominant_freq_index = np.argmax(fft_data) 
    dominant_freq = x[dominant_freq_index]  
    

    if dominant_freq < 100:
        line.set_color('red') 
    elif 100 <= dominant_freq < 400:
        line.set_color('orange') 
    elif 400 <= dominant_freq < 1000:
        line.set_color('yellow') 
    elif 1000 <= dominant_freq < 5000:
        line.set_color('blue')  
    else:
        line.set_color('violet')  

    for i, ball in enumerate(balls):
        y_pos = fft_data[i] / max(fft_data) * 200  
        ball.set_center((x[i], y_pos))  

        ball.set_radius(5 + (y_pos / 200) * 20)  

        if x[i] < 100:  
            color = 'black'
        elif 100 <= x[i] < 500:  
            color = 'orange'
        elif 500 <= x[i] < 3000: 
            color = 'yellow'
        elif 3000 <= x[i] < 5000:  
            color = 'blue'
        else:  
            color = 'violet'

        ball.set_facecolor(color)

    return line, *balls




fig, ax = plt.subplots()
x = np.linspace(0, RATE / 2, CHUNK // 2)  
line, = ax.plot(x, np.zeros(CHUNK // 2)) 
ax.set_ylim(0, 300)  
ax.set_xlim(0, RATE // 2)  
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")


i = 0
balls = [plt.Circle((x[i], 0), 5, color='grey') for i in range(CHUNK // 2)]
for ball in balls:
    ax.add_patch(ball)

ani = FuncAnimation(fig, update_plot, blit=True, interval=50)

root = tk.Tk()
root.title("Emotion-Based Song Player")

capture_button = tk.Button(root, text="Capture Image", command=capture_and_predict)
capture_button.pack(pady=10)

fft_canvas = tk.Canvas(root, width=400, height=200, bg='white')
fft_canvas.pack(pady=20)

root.protocol("WM_DELETE_WINDOW", lambda: root.quit())

plt.show()

root.mainloop()

stream.stop_stream()
stream.close()
p.terminate()
