import pygame
from camera import Camera, Follow, Auto
from vehicle import Car

from map import Town01

robot_img_path = (
    "/home/dan/Data/projects/reinforcement/carla-bev-env/CarlaBEV/envs/robot-gr.png"
)

# msi
map_path = "/home/dan/Data/datasets/CarlaBEV/Town01-1024-RGB.jpg"

# home
# map_path = "/home/danielmtz/Data/datasets/CarlaBEV/maps/Town01/Town01-1024-RGB.jpg"
theta = 0


class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.img = pygame.image.load(robot_img_path)
        self.rect = self.img.get_rect()

        self.box = pygame.Rect(self.rect.x, self.rect.y, self.rect.w * 2, self.rect.h)
        self.box.center = self.rect.center

        # controls
        self.LEFT_KEY, self.RIGHT_KEY = False, False
        self.UP_KEY, self.DOWN_KEY = False, False
        #
        self.vx, self.vy = 0, 0
        self.theta = 0

    def update(self):
        self.vx, self.vy = 0, 0
        if self.UP_KEY:
            self.vy = 5
        elif self.DOWN_KEY:
            self.vy = -5
        if self.LEFT_KEY:
            self.vx = 5
        elif self.RIGHT_KEY:
            self.vx = -5
        self.rect.x += self.vx
        self.rect.y += self.vy
        print(self.rect.x, self.rect.y)

    def draw(self, display, pos=(512, 512), originPos=(0, 0)):
        # offset from pivot to center
        image_rect = self.img.get_rect(
            topleft=(pos[0] - originPos[0], pos[1] - originPos[1])
        )
        offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center

        # roatated offset from pivot to center
        rotated_offset = offset_center_to_pivot.rotate(-self.theta)

        # roatetd image center
        rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

        # get a rotated image
        rotated_image = pygame.transform.rotate(self.img, self.theta)
        rotated_image_rect = rotated_image.get_rect(center=rotated_image_center)

        # rotate and blit the image
        display.blit(rotated_image, rotated_image_rect)


################################# LOAD UP A BASIC WINDOW AND CLOCK #################################
pygame.init()
DISPLAY_W, DISPLAY_H = 1024, 1024
canvas = pygame.Surface((6144, 8192))
window = pygame.display.set_mode(((DISPLAY_W, DISPLAY_H)))
running = True
clock = pygame.time.Clock()
# map = pygame.image.load(map_path).convert()
map = Town01(window_size=(DISPLAY_H, DISPLAY_W))

################################# LOAD PLAYER AND CAMERA###################################
# car = Car(spawn_position=[500, 500, 0], length=1)
car = Player()
camera = Camera(car, resolution=(DISPLAY_W, DISPLAY_H))
follow = Follow(camera, car)
auto = Auto(camera, car)
camera.setmethod(follow)
################################# GAME LOOP ##########################
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

    theta += 1
    map.set_theta(theta)

    ################################# UPDATE/ Animate SPRITE #################################
    car.update()
    camera.scroll()
    ################################# UPDATE WINDOW AND DISPLAY #################################

    # map.blitRotate(canvas, (camera.offset.x, camera.offset.y), originPos=map.origin)
    map.blit_fov(canvas, (camera.offset.x, camera.offset.y))
    car.draw(canvas, pos=(512, 512))

    window.blit(canvas, (0, 0))
    pygame.display.update()
