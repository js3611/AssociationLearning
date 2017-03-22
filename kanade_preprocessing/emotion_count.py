import os
import numpy as np

print 'Counting number of files'

os.chdir('Emotion')

num_emotions = np.zeros(7)
for subject in os.listdir('.'):
    for emo in os.listdir(subject):
        for file in os.listdir(os.path.join(subject, emo)):
            if file.endswith('.txt'):
                with open(os.path.join(subject, emo, file),'r') as f:
                    label = f.readline().strip().split('.')[0]
                    idx = int(label)
                    num_emotions[idx-1] += 1

print num_emotions
            
