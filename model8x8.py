import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

import asyncio
import monome
import pygame 
import random
import vlc

pygame.init()
clock = pygame.time.Clock()

def rythm(loop):
     # KI-Modell
    model = Sequential()
    model.add(LSTM(128, input_shape=(8, 8)))
    model.add(Dense(64, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Generierung von Rhythmen
    start_pattern = np.zeros((8, 8))
    start_pattern = np.reshape(start_pattern, (1, 8, 8))  # Fügt eine Dimension hinzu

    for i in range(5):
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

    loop.stop()


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
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    grid_studies = GridStudies()

    def serialosc_device_added(id, type, port):
        print('connecting to {} ({})'.format(id, type))
        asyncio.ensure_future(grid_studies.grid.connect('127.0.0.1', port))

    serialosc = monome.SerialOsc()
    serialosc.device_added_event.add_handler(serialosc_device_added)

    loop.run_until_complete(serialosc.connect())
    loop.call_soon(rythm,loop);
    loop.run_forever()

    for r in rythmA:
        x = 0
        y = 0
        for row in r:
            for element in row:
                x = round(x/(y+1))
                print(element,x,y)
                if(element==1):
                    grid_studies.on_grid_key(x,y,1)
                    p = vlc.MediaPlayer("file:///home/erhard/code/python/monomegrid/2269__jobro__laser-shots/"+str(x+1)+".wav")
                    p.play()
                    #clock.tick()
                x+=1
            y+=1
        
        resetGrid();
    #print(rythmA);

