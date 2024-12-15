import pygame
from camera import Camera, Follow, Auto
from vehicle import Car, Player

from map import Town01

robot_img_path = (
    "/home/danielmtz/Data/projects/carla-bev-env/CarlaBEV/envs/robot-gr.png"
)

# msi
map_path = "/home/dan/Data/datasets/CarlaBEV/Town01-1024-RGB.jpg"

# home
# map_path = "/home/danielmtz/Data/datasets/CarlaBEV/maps/Town01/Town01-1024-RGB.jpg"
theta = 0


################################# LOAD UP A BASIC WINDOW AND CLOCK #################################
pygame.init()
DISPLAY_W, DISPLAY_H = 1024, 1024
canvas = pygame.Surface((7168, 9216))
window = pygame.display.set_mode(((DISPLAY_W, DISPLAY_H)))
clock = pygame.time.Clock()

################################# LOAD PLAYER AND CAMERA###################################
# car = Car(spawn_position=[500, 500, 0], length=1)
map = Town01(window_size=(DISPLAY_H, DISPLAY_W))
car = Player()
camera = Camera(car, resolution=(DISPLAY_W, DISPLAY_H))
follow = Follow(camera, car)
auto = Auto(camera, car)
camera.setmethod(follow)
################################# GAME LOOP ##########################
running = True
while running:
    clock.tick(60)
    ################################# CHECK PLAYER INPUT #################################
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                car.LEFT_KEY = True
            elif event.key == pygame.K_RIGHT:
                car.RIGHT_KEY = True
            elif event.key == pygame.K_UP:
                car.UP_KEY = True
            elif event.key == pygame.K_DOWN:
                car.DOWN_KEY = True
            elif event.key == pygame.K_1:
                camera.setmethod(follow)
            elif event.key == pygame.K_2:
                camera.setmethod(auto)

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                car.LEFT_KEY = False
            elif event.key == pygame.K_RIGHT:
                car.RIGHT_KEY = False
            elif event.key == pygame.K_UP:
                car.UP_KEY = False
            elif event.key == pygame.K_DOWN:
                car.DOWN_KEY = False

    ################################# UPDATE/ Animate SPRITE #################################
    car.update()
    map.set_theta(car.theta)
    camera.scroll()
    ################################# UPDATE WINDOW AND DISPLAY #################################

    map.blitRotate(canvas, topleft=camera.offset, pos=(0, 0))
    car.draw(canvas, pos=(512, 512))

    window.blit(canvas, (0, 0))
    pygame.display.update()
