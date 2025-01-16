import pygame
from camera import Camera, Follow, Auto
from vehicle import Car

from map import Town01

robot_img_path = (
    "/home/danielmtz/Data/projects/carla-bev-env/CarlaBEV/envs/robot-gr.png"
)

################################# LOAD UP A BASIC WINDOW AND CLOCK #################################
pygame.init()
DISPLAY_W, DISPLAY_H = 1024, 1024
window = pygame.display.set_mode(((DISPLAY_W, DISPLAY_H)))
clock = pygame.time.Clock()

################################# LOAD PLAYER AND CAMERA###################################
id = 0
map = Town01(window_size=(DISPLAY_H, DISPLAY_W), target_id=id, size=128)
car = Car()
camera = Camera(car, resolution=(DISPLAY_W, DISPLAY_H))
follow = Follow(camera, car)
camera.setmethod(follow)
################################# GAME LOOP ##########################
running = True
while running:
    clock.tick(60)
    action = 0
    ################################# CHECK PLAYER INPUT #################################
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 1
            elif event.key == pygame.K_RIGHT:
                action = 2
            elif event.key == pygame.K_UP:
                action = 3
            elif event.key == pygame.K_DOWN:
                action = 4

    ################################# UPDATE/ Animate SPRITE #################################
    car.update(action)
    map.set_theta(car.theta)
    camera.scroll()
    ################################# UPDATE WINDOW AND DISPLAY #################################

    map.blitRotate(window, topleft=camera.offset, pos=(0, 0))
    print(map.agent_tile)
    car.draw(window, pos=(512, 512))

    window.blit(pygame.transform.rotate(window, 90), (0, 0))
    pygame.display.update()
