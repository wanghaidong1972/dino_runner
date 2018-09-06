# coding: utf-8

import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from PIL import Image
from io import BytesIO
import base64
import cv2

import numpy as np
import copy
from collections import deque

COL_SIZE = 80
ROW_SIZE = 80

def show_img():
    while True:
        img = (yield)
        window_title = "playing"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        img = cv2.resize(img, (80, 80))
        cv2.imshow(window_title, img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break


def process_img(image):
    image = cv2.resize(image, (COL_SIZE, ROW_SIZE))
    image = image[:300, :500]  # Crop ROI
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    return image

class DinoEnv:

    def __init__(self, driver="chromedriver.exe"):

        if not os.path.exists(driver):
            raise IOError('driver path not found{}'.format(driver))

        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path=driver, chrome_options=chrome_options)
        self._driver.set_window_position(x=10, y=10)
        self._driver.get("chrome://dino")
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")
        self._display = show_img()  # show processed image by python coroutine
        self._display.__next__()

        self._action_set = [0, 1]  # keep stay or jump
        self._jump() # start the game

        img_0 = self._get_image()

        self._image_stack = np.stack((img_0, img_0, img_0, img_0), axis=2)  # initialize stack
        self._image_stack = self._image_stack.reshape(1,
                                                      COL_SIZE,
                                                      ROW_SIZE,
                                                      -1)
        # self._image_stack = deque(maxlen=STACK_SIZE)
        # for i in range(STACK_SIZE):
            # self._image_stack.append(img_0)

        self._initial_stack = copy.deepcopy(self._image_stack)

        # self._pause()

    def get_current_status(self):
        return self._image_stack

    def get_ini_status(self):
        return self._initial_stack

    def step(self, a):
        reward = 0.1
        action = self._action_set[a]

        # self._resume()
        if action == 1:  # jump
            self._jump()
        else:
            pass
        ob = self._get_obs()
        ob = ob.reshape(1,COL_SIZE, ROW_SIZE,1)

        self._image_stack = np.append(ob, self._image_stack[:, :, :, :3], axis=3)

        gameover = False

        if self._is_over():
            gameover = True
            reward = -1
            self._restart()

        # self._pause()

        return self._image_stack, reward, gameover, {}

    def _pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def pause(self):
        self._pause()

    def _resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def resume(self):
        self._resume()

    def _restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
        self._image_stack = self._initial_stack

    def _jump(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def _is_over(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def _get_image(self):
        image_b64 = self._driver.execute_script(
            "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)")
        hard_copy = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        image = process_img(hard_copy)  # processing image as required

        self._display.send(image)

        return image


    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        img = self._get_image()
        return img

    # return: (states, observations)
    def reset(self):
        self._restart()
        return self._image_stack

    def close(self):
        self._driver.close()
        cv2.destroyAllWindows()


    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        # we only play robotic game(no human) ,so just leave this method blank
        keys_to_action = {}
        return keys_to_action

    def clone_state(self):
        """leave it blank on this time"""
        return None

    def restore_state(self, state):
        """leave it blank on this time"""
        pass

ACTION_MEANING = {
    0 : "NOOP",
    1 : "JUMP",
}
