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

You can also build a Docker environment by the dockerfile in this repository.  the commands are like following.

1.  build image : build  -t dino_run:v1 -f Dockerfile  .
2.  start container: docker run -p 5901:5901 -p 6901:6901 dino_run:v1
3.  connect to container :  use a vnc viewer and connect to  127.0.0.1:5901 or 
a HTML5 client(like Firefox) to http://127.0.0.1:6901/vnc.html  . the password is vncpassword
4.  open a term from vnc viewer , change directory to /headleass/dino/dino_runner and type python runner.py

Attention : When train in docker , the result will not be as good as expected due to low FPS, it's more like an example for explaning how to build an environment . (anyway,this image does not even remove downloaded files)
