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

selenium_webdriver = None

class MetacarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, level, discrete):

        self.level = level
        self.discrete = discrete

        # Observation space.
        if self.discrete == True:
            lidar_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.uint8)
            linear_space = spaces.Box(low=-1, high=1, shape=(26,), dtype=np.uint8)
            v_space = spaces.Box(low=-1, high=1, shape=(1,))
            self.observation_space = spaces.Dict({"lidar": lidar_space, "linear": linear_space, "v": v_space})
        else:
            lidar_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.uint8)
            linear_space = spaces.Box(low=-1, high=1, shape=(26,), dtype=np.uint8)
            a_space = spaces.Box(low=-1, high=1, shape=(1,))
            v_space = spaces.Box(low=-1, high=1, shape=(1,))
            steering_space = spaces.Box(low=-1, high=1, shape=(1,))
            self.observation_space = spaces.Dict({"lidar": lidar_space, "linear": linear_space, "a": v_space, "v": v_space, "steering": v_space})

        # Action space.
        if self.discrete == True:
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Prepare for pygame.
        self._pygame_screen = None

        # Prepare for web rendering.
        self.webrenderer = False

    def enable_webrenderer(self):
        self.webrenderer = True

    def step(self, action):
        reward = selenium_webdriver.execute_script(f"return env.step({action});")
        observation = selenium_webdriver.execute_script(f"return env.getState();")
        done = False
        info = {}
        return observation, reward, done, info

    def reset(self):

        global selenium_webdriver

        # Close webdriver.
        if selenium_webdriver != None:
            selenium_webdriver.quit()

        # Enable webdriver.
        print("Creating web driver...")
        options = Options()
        if self.webrenderer == False:
            options.headless = True
            options.add_argument('window-size=800x800') # optional
        else:
            options.headless = False
        options.add_argument('no-sandbox')
        options.add_argument('disable-dev-shm-usage')
        options.add_argument("disable-infobars");
        selenium_webdriver = webdriver.Chrome("chromedriver", options=options)
        print("Created web driver.")

        # Load the web page.
        html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "metacar.html")
        url = "file://" + html_file
        selenium_webdriver.get(url)
        delay = 3 # seconds
        try:
            _ = WebDriverWait(selenium_webdriver, delay).until(EC.presence_of_element_located((By.ID, "canvas")))
        except TimeoutException:
            raise Exception("Loading took too much time!")

        # Trigger environment initialization.
        script = ""
        script += 'document.getElementById("canvas").style.visibility = "hidden"' + "\n"
        script += 'let levelUrl = metacar.level.{}'.format(self.level) + "\n"
        script += 'env = new metacar.env("canvas", levelUrl);' + "\n"
        if self.discrete == True:
            script += 'env.setAgentMotion(metacar.motion.BasicMotion, {rotationStep: 0.25});' + "\n"
        else:
            script += 'env.setAgentMotion(metacar.motion.ControlMotion);' + "\n"
        script += 'env.load().then(() => { document.getElementById("canvas").style.visibility = "visible"; });' + "\n"
        selenium_webdriver.execute_script(script)

        # Wait for the environment to initialize.
        delay = 10
        try:
            _ = WebDriverWait(selenium_webdriver, delay).until(EC.visibility_of_element_located((By.ID, "canvas")))
        except TimeoutException:
            raise Exception("ERROR! Initializing environment took too much time!")

        observation = selenium_webdriver.execute_script(f"return env.getState();")
        return observation


    def render(self, mode='human', close=False):

        # Update web renderer.
        if self.webrenderer == True:
            script = 'displayState("lidar", env.getState().lidar, 200, 200);'
            selenium_webdriver.execute_script(script)

        # Render with pygame.
        else:
            # Get the canvas element.
            element = selenium_webdriver.find_element_by_id("canvas").find_elements_by_css_selector("*")[0]

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
        pygame.quit()


# Close the driver in the end
def metacar_env_exit():
    print("Thank you for playing.")
    selenium_webdriver.quit()
atexit.register(metacar_env_exit)
