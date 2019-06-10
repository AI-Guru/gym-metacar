import gym
from gym import error, spaces, utils
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import os
from PIL import Image
from io import BytesIO
import pygame
import numpy as np
import atexit
import random


# Uses a selenium webdriver under the hood
selenium_webdriver = None


class MetacarEnv(gym.Env):
    """
    OpenAI Gym wrapper for Metacar: A reinforcement learning environment for self-driving cars in the browser.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, level, discrete):

        # Store parameters.
        self.level = level
        self.discrete = discrete

        # Observation space.
        if self.discrete == True:
            lidar_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.float32)
            linear_space = spaces.Box(low=-1, high=1, shape=(26,), dtype=np.float32)
            v_space = spaces.Box(low=-1, high=1, shape=(1,))
            self.observation_space = spaces.Dict({"lidar": lidar_space, "linear": linear_space, "v": v_space})
        else:
            lidar_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.float32)
            linear_space = spaces.Box(low=-1, high=1, shape=(26,), dtype=np.float32)
            a_space = spaces.Box(low=-1, high=1, shape=(1,))
            v_space = spaces.Box(low=-1, high=1, shape=(1,))
            steering_space = spaces.Box(low=-1, high=1, shape=(1,))
            self.observation_space = spaces.Dict({"lidar": lidar_space, "linear": linear_space, "a": v_space, "v": v_space, "steering": v_space})

        # Action space.
        if self.discrete == True:
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Prepare for pygame.
        self._pygame_screen = None

        # Prepare for web rendering.
        self.webrenderer = False

        # The environment should be reset before use.
        self.has_been_reset = False


    def enable_webrenderer(self):
        """
        Enables the web-renderer. Disables pygame.
        """

        self.webrenderer = True


    def step(self, action):
        """
        Performs one step in the environment.
        """

        if self.has_been_reset == False:
            raise Exception("ERROR! You have to reset the environment.")

        # Execute one step and get the reward.
        if self.discrete == True:
            reward = selenium_webdriver.execute_script("return env.step({});".format(action))
        else:
            action = "[" + ", ".join([str(a) for a in action]) + "]"
            reward = selenium_webdriver.execute_script("return env.step({});".format(action))

        # Get the observation from the environment.
        observation = selenium_webdriver.execute_script("return env.getState();")

        # So far, simulations do not end.
        done = False

        # No info to provide.
        info = {}

        return observation, reward, done, info


    def reset(self):
        """
        Resets the environment
        """

        global selenium_webdriver

        # Lazy enable webdriver.
        if selenium_webdriver == None:
            print("Creating web driver...")
            options = Options()
            if self.webrenderer == False:
                options.headless = True
                options.add_argument('window-size=800x800')
            else:
                options.headless = False
            options.add_argument('no-sandbox')
            options.add_argument('disable-dev-shm-usage')
            options.add_argument("disable-infobars");
            selenium_webdriver = webdriver.Chrome("chromedriver", options=options)
            print("Created web driver.")

        # Load the underlying web page.
        html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "metacar.html")
        url = "file://" + html_file
        selenium_webdriver.get(url)
        delay = 3 # seconds
        try:
            _ = WebDriverWait(selenium_webdriver, delay).until(EC.presence_of_element_located((By.ID, "canvas")))
        except TimeoutException:
            raise Exception("ERROR! Could not load unterlying web page")

        # Trigger environment initialization.
        if self.level == "random":
            level_string = random.choice(["level0", "level1", "level2", "level3"])
        else:
            level_string = self.level
        script = ""
        script += 'document.getElementById("canvas").style.visibility = "hidden"' + "\n"
        script += 'let levelUrl = metacar.level.{}'.format(level_string) + "\n"
        script += 'env = new metacar.env("canvas", levelUrl);' + "\n"
        if self.discrete == True:
            script += 'env.setAgentMotion(metacar.motion.BasicMotion, {rotationStep: 0.25});' + "\n"
        else:
            script += 'env.setAgentMotion(metacar.motion.ControlMotion, {});' + "\n"
        script += 'env.load().then(() => { document.getElementById("canvas").style.visibility = "visible"; env.shuffle({cars: false}); });' + "\n"
        selenium_webdriver.execute_script(script)

        # Wait for the environment to initialize.
        delay = 10
        try:
            _ = WebDriverWait(selenium_webdriver, delay).until(EC.visibility_of_element_located((By.ID, "canvas")))
        except TimeoutException:
            raise Exception("ERROR! Could not initialize the environment in time.")

        # Success!
        self.has_been_reset = True

        # Yield the first observation.
        observation = selenium_webdriver.execute_script("return env.getState();")
        return observation


    def render(self, mode='human', close=False):
        """
        Renders the current scene.
        """

        # Update web renderer.
        if self.webrenderer == True:
            script = 'displayState("lidar", env.getState().lidar, 200, 200);'
            selenium_webdriver.execute_script(script)

        # Render with pygame.
        else:
            # Get the canvas element.
            element = selenium_webdriver.find_element_by_id("canvas").find_elements_by_css_selector("*")[0]

            # Get pixel ration to deal with retina/no-retina displays.
            device_pixel_ratio = selenium_webdriver.execute_script("return window.devicePixelRatio")

            # Lazy loading pygame.
            if self._pygame_screen == None:
                self.canvas_left = element.location['x']
                self.canvas_top = element.location['y']
                self.canvas_right = element.location['x'] + element.size['width']
                self.canvas_bottom = element.location['y'] + element.size['height']

                pygame.init()
                pygame.display.set_caption("gym_metacar")
                self.screen_width = element.size["width"]
                self.screen_height = element.size["height"]
                self._pygame_screen = pygame.display.set_mode((self.screen_width, self.screen_height))

            # Consume events.
            for event in pygame.event.get():
                pass

            # Render.
            image = selenium_webdriver.get_screenshot_as_png()
            pil_image = Image.open(BytesIO(image))
            pil_image = pil_image.crop((self.canvas_left * device_pixel_ratio, self.canvas_top * device_pixel_ratio, self.canvas_right * device_pixel_ratio, self.canvas_bottom * device_pixel_ratio)) # defines crop points
            pil_image = pil_image.resize((self.screen_width, self.screen_height), Image.ANTIALIAS)
            pil_image = np.array(pil_image)
            pil_image = pil_image[:,:,0:3]
            pil_image = np.swapaxes(pil_image, 0, 1)
            surface = pygame.surfarray.make_surface(pil_image)
            self._pygame_screen.blit(surface, (0, 0))

            # Flip
            pygame.display.flip()


    def close(self):
        """
        Closes the environment.
        """

        pygame.quit()


# Close the driver in the end
def metacar_env_exit():
    print("Thank you for playing.")
    selenium_webdriver.quit()
atexit.register(metacar_env_exit)
