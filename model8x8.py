import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

import asyncio
import monome
import pygame 
import pyaudio

# Define some parameters
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sample rate

# Create pyaudio object
p = pyaudio.PyAudio()

# Open a stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True)

# Generate a sine wave
def generate_sine_wave(frequency, duration):
  omega = 2 * np.pi * frequency
  samples = (np.sin(omega * np.linspace(0, duration, int(RATE * duration))) * 32767).astype(np.int16)
  return samples.tobytes()

def generate_square_wave(frequency, duration):
  omega = 2 * np.pi * frequency
  duty_cycle = 0.5  # Adjust for pulse width
  samples = np.where(np.linspace(0, duration, int(RATE * duration)) < duty_cycle * duration, 1, -1) * 32767
  return samples.tobytes()


def generate_triangle_wave(frequency, duration):
  omega = 2 * np.pi * frequency
  t = np.linspace(0, duration, int(RATE * duration))
  samples = (2 * np.abs(t - duration / 2) / duration - 1) * 32767
  return samples.tobytes()


def generate_white_noise(duration):
  samples = np.random.rand(int(RATE * duration)) * 32767
  return samples.astype(np.int16).tobytes()

def generate_sawtooth_wave(frequency, duration):
  omega = 2 * np.pi * frequency
  t = np.linspace(0, duration, int(RATE * duration))
  samples = (t / duration) * 32767
  return samples.tobytes()



pygame.init()
clock = pygame.time.Clock()

def sound(loop):    
    # KI-Modell
    model = Sequential()
    model.add(LSTM(128, input_shape=(8, 8)))
    model.add(Dense(64, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Generierung von Rhythmen
    start_pattern = np.zeros((8, 8))
    start_pattern = np.reshape(start_pattern, (1, 8, 8))  # Fügt eine Dimension hinzu

    for i in range(10):
        # Vorhersage
        prediction = model.predict(start_pattern)
        # Aktualisierung des Bretts
        start_pattern = np.roll(start_pattern, 1, axis=1)
        start_pattern[:, -1] = prediction[:, :8] 
        
        # Visualisierung
        # ...
        threshold = 0.01562
        data_binary = np.where(prediction > threshold, 1, 0)
        #print(data_binary)
        array_2d = np.reshape(data_binary, (8, 8))
        rythmA.append(array_2d)
    
    
    while True:
        master = 0
        for x in rythmA:
            for i in range(0,8):
                for k in range(0,8):                    
                    grid_studies.on_grid_key(k,0,1) 
                    if x[k][i] == 1:                                      #   k = 1, i = 1
                        grid_studies.on_grid_key(k,i,1)                        
                        #data = generate_sine_wave(440, 0.1)                        
                        data = generate_sine_wave(440+(10*(i+1)), 0.05)
                        stream.write(data)
                        if master % 2 == 0:
                            data = generate_triangle_wave(440+(10*(i+1)), 0.001)
                            stream.write(data)
                            data = generate_sawtooth_wave(440+(10*(i+1)), 0.0005)
                            stream.write(data)
                    clock.tick(10) 
                    resetGrid() 
            master += 1
    # Stop the stream and close PyAudio
    
    stream.stop_stream()
    stream.close()
    p.terminate()
            
    
    

class GridStudies(monome.GridApp):
    def __init__(self):
        super().__init__()

    def on_grid_key(self, x, y, s):
        #print("key:", x, y, s)
        self.grid.led_level_set(x, y, s * 15)

def resetGrid():
    for i in range(0,8):
        for k in range(0,8):
            grid_studies.on_grid_key(i,k,0);


grid_studies = ''
rythmA = []
"""
         [[[1,0,0,0,0,0,0,0],
           [0,1,0,0,0,0,0,0],
           [0,0,1,0,0,0,0,0],
           [0,0,0,1,1,0,0,0],
           [0,0,0,0,0,1,1,0],
           [0,0,0,0,0,0,1,0],
           [0,0,0,0,0,0,0,1],
           [0,0,0,0,0,0,1,1]]]
"""
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    grid_studies = GridStudies()

    def serialosc_device_added(id, type, port):
        print('connecting to {} ({})'.format(id, type))
        asyncio.ensure_future(grid_studies.grid.connect('127.0.0.1', port))

    serialosc = monome.SerialOsc()
    serialosc.device_added_event.add_handler(serialosc_device_added)

    loop.run_until_complete(serialosc.connect())
    loop.call_soon(sound,loop)
    loop.run_forever()
