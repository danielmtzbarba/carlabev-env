
import pygame
import sys

import numpy as np
import math

from CarlaBEV.utils import load_map

pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLUE = (0, 120, 215)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BUTTON_COLOR = (50, 150, 50)
BUTTON_HOVER = (70, 180, 70)

# Initialize window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic Scenario Designer")
font = pygame.font.SysFont(None, 24)

# Actor data structure
actors = []
current_start = None
adding_actor = False

# Button
button_rect = pygame.Rect(10, 10, 140, 30)



class Town01(object):
    def __init__(self, size=1024) -> None:
        self._map_arr, self._map_img, _ = load_map(size)
        self._Y, self._X, _ = self._map_arr.shape
        self.size = size  # The size of the square grid
        self.center = (int(self.size / 2), int(self.size / 2))
        self._map_surface = pygame.Surface((self._X, self._Y))
        self._fov_surface = pygame.Surface((self.size, self.size))
        #
        self._map_surface.blit(self._map_img, (0, 0))
        self._scene = Scene(map_surface=self._map_surface, size=self.size)
        #

def draw_map():
    screen.fill(GREY)
    # Placeholder for actual map drawing (could load image or draw lines, etc.)
    pygame.draw.rect(screen, WHITE, (0, 50, WIDTH, HEIGHT - 50))


def draw_button():
    mouse_pos = pygame.mouse.get_pos()
    color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, color, button_rect)
    text = font.render("Add Actor", True, WHITE)
    screen.blit(text, (button_rect.x + 20, button_rect.y + 5))


def draw_actors():
    for actor in actors:
        pygame.draw.circle(screen, BLUE, actor['start'], 6)
        pygame.draw.circle(screen, RED, actor['end'], 6)
        pygame.draw.line(screen, BLUE, actor['start'], actor['end'], 2)


# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                adding_actor = True
                current_start = None
            elif adding_actor:
                if current_start is None:
                    current_start = event.pos
                else:
                    actors.append({'start': current_start, 'end': event.pos})
                    current_start = None
                    adding_actor = False

    draw_map()
    draw_button()
    draw_actors()
    pygame.display.flip()
