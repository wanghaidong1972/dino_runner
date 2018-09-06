# dino_runner

There's an Easter egg of chrome. Yes,when be offline,you can play a dinosure runner game.
This program try to play this game by reinforcement learning.

Requirement: Keras,OpenCV,Selenium,Pillow. 

Also , you should put chrome driver which you can find from [this site](http://chromedriver.chromium.org) under the same directory.

Now, run python runner.py and enjoy the training.

You can quit traing any time by typing 'q' in the 'palying' window which will show the 
resized images dynamically.

You can continue the training from the breakpoint since this program save them every
3000 steps.

Of course there are still some tunning can be done,but by now it can achieve a 
decent score like 300 by 2 hours training on a modern PC even without GPU.