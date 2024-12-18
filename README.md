# Emotion-Based Song Player with Real-Time FFT Visualization

This project captures an image using the webcam, predicts the user's emotion using a pre-trained deep learning model, and plays a corresponding song based on the detected emotion. It also provides a real-time audio frequency visualization using Fast Fourier Transform (FFT) with colorful, dynamic balls to enhance the user experience.

## Features
1. **Emotion Detection**
   - Captures an image using the webcam.
   - Processes the image using a pre-trained TensorFlow/Keras model.
   - Detects one of the following emotions:
     - Angry
     - Fearful
     - Happy
     - Sad
     - Surprised
     - Disgusted
     - Neutral

2. **Song Playback Based on Emotion**
   - Plays a random song from a directory specific to the detected emotion.
   - Supports MP3 files using the default media player.

3. **Real-Time Audio FFT Visualization**
   - Captures microphone input and visualizes the frequency spectrum using FFT.
   - Uses dynamic colors to represent different frequency bands.
   - Floating balls move and resize based on the amplitude of frequencies.

4. **GUI Interface**
   - Simple GUI using Tkinter for capturing images and starting the process.
   - Integrated with Matplotlib for live visualization.

---
## Directory Structure
The following folder structure is assumed for the project:

```
project_root/
|-- model/
|   |-- model.h5                # Pre-trained emotion detection model
|
|-- songs/
|   |-- Angry/                  # Songs for 'Angry' emotion
|   |-- Fearful/                # Songs for 'Fearful' emotion
|   |-- Happy/                  # Songs for 'Happy' emotion
|   |-- Sad/                    # Songs for 'Sad' emotion
|   |-- Surprised/              # Songs for 'Surprised' emotion
|   |-- Disgusted/              # Songs for 'Disgusted' emotion
|   |-- Neutral/                # Songs for 'Neutral' emotion
|
|-- app.py                      # Main project file
|-- README.md                   # Project documentation
```

---
## Requirements
Ensure the following dependencies are installed before running the project:

### Python Libraries
- **TensorFlow**: For emotion prediction using the deep learning model.
- **OpenCV**: For image capture and preprocessing.
- **NumPy**: For numerical operations.
- **pyaudio**: For real-time audio input.
- **Matplotlib**: For live FFT visualization.
- **pydub**: For audio processing (if needed).
- **tkinter**: Built-in Python GUI library.

Install the required libraries using pip:
```bash
pip install tensorflow opencv-python-headless numpy pyaudio matplotlib pydub
```

---
## Usage
1. **Setup Emotion Model**:
   - Place your pre-trained `model.h5` file in the `model` directory.
   - Ensure the model is trained to classify the 7 emotions: *Angry, Fearful, Happy, Sad, Surprised, Disgusted, Neutral*.

2. **Add Songs**:
   - Organize MP3 songs into folders under the `songs` directory. Each folder should match an emotion name.
   - Example:
     ```
     songs/
     |-- Happy/
     |   |-- song1.mp3
     |   |-- song2.mp3
     |-- Sad/
         |-- song3.mp3
     ```

3. **Run the Application**:
   - Execute the Python script:
     ```bash
     python app.py
     ```

4. **Using the Application**:
   - Click on the **"Capture Image"** button to take a photo and predict your emotion.
   - Based on the prediction, a song will play from the corresponding folder.
   - Observe the real-time FFT visualization in a separate window.

---
## How It Works
1. **Emotion Prediction**:
   - Captures a frame from the webcam using OpenCV.
   - Preprocesses the image (grayscale, resizing, normalization).
   - Feeds the image into a pre-trained TensorFlow model.
   - Determines the predicted emotion using `np.argmax()`.

2. **Song Selection**:
   - Retrieves all songs from the folder corresponding to the detected emotion.
   - Plays a random song using the system's default media player.

3. **Real-Time FFT Visualization**:
   - Captures audio input using `pyaudio`.
   - Applies Fast Fourier Transform (FFT) to the audio data.
   - Displays the frequency spectrum in real-time.
   - Updates the position, size, and color of dynamic balls based on frequency amplitude.

4. **GUI Interface**:
   - Built using Tkinter.
   - Provides a simple button to capture images and initiate the process.

---
## Real-Time FFT Visualization Details
- **X-Axis**: Frequency in Hz (up to Nyquist frequency, `RATE / 2`).
- **Y-Axis**: Amplitude of frequencies.
- **Colors**:
  - Low frequencies (0-100 Hz): **Red**
  - Mid-low frequencies (100-400 Hz): **Orange**
  - Mid frequencies (400-1000 Hz): **Yellow**
  - High frequencies (1000-5000 Hz): **Blue**
  - Very high frequencies (>5000 Hz): **Violet**

---
## Notes
- Ensure your microphone and webcam are working.
- Songs should be in MP3 format and placed in the correct folders.
- The model must be compatible with input shape `(1, 48, 48, 1)` (grayscale, 48x48 images).

---
## Future Improvements
- Add support for streaming audio directly within the application.
- Integrate a media player within the GUI for better song controls.
- Expand the emotion model to support more nuanced emotions.
- Improve the FFT visualization for a smoother user experience.

---
## License
This project is for educational purposes. Modify and distribute as needed.

---
## Author
**Ashish Pandey**  
Feel free to contact for suggestions, feedback, or collaboration!
