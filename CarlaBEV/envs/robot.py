import math
import pygame
import os
import utils

Projectfolder_image = "assets/"

Body_Robot = os.path.join(Projectfolder_image, "Body.png")
Wheel_Robot = os.path.join(Projectfolder_image, "wheel.png")


class Robot(pygame.sprite.Sprite):
    m2p = 3779.52  # meters 2 pixels

    def __init__(self, startpos, window_center, size, color=(0, 7, 175), car_size=32):
        pygame.sprite.Sprite.__init__(self)
        self.color = color
        self.size = size
        self.scale = int((1024 / size))
        self._window_center = window_center
        self._w = int(car_size / self.scale)
        self._l = 2 * self._w
        # self.xpath, self.ypath = path
        self.x, self.y = startpos
        #

        self._setup()

    def _setup(self):
        self._spawn_location = (
            int(self.x - self._l / 2),
            int(self.y - self._w / 2),
        )
        #
        self._draw_rect = pygame.Rect(
            self._window_center[0] - int(self._l / 2),
            self._window_center[1] - int(self._l / 2),
            self._w,
            self._l,
        )
        self.rect = pygame.Rect(
            self._spawn_location[0], self._spawn_location[1], self._w, self._l
        )

        def setup_kinematics(self):
            # kinematics
            self.w = 0  # rad/sec
            self.theta = 5
            self.psi = 0  # rad/s
            self.a = 20
            self.lF = 30
            self.lR = 30
            self.beta = 0
            self.r = 0

            # wheels
            self.xF = self.x + self.lF * math.cos(self.psi)
            self.yF = self.y + self.lF * math.sin(self.psi)
            self.xR = self.x - self.lR * math.cos(self.psi)
            self.yR = self.y - self.lR * math.sin(self.psi)
            #
            self.beta = 0
            self.r = 0
            self.u = 180  # pix/sec
            self.x_back = 0
            self.y_back = 0
            self.maxspeed = 0.02 * self.m2p
            self.minspeed = -0.02 * self.m2p

        def setup_graphics(self):
            # graphics
            self.img = pygame.image.load(Body_Robot)
            # Scale the image to your needed size
            self.img = pygame.transform.scale(self.img, (self._l, self._w))
            self.rotated = self.img
            self.rect = self.rotated.get_rect(center=(self.x, self.y))

            self.wheelF = pygame.image.load(Wheel_Robot)
            self.rotated_wheelF = self.wheelF
            self.rect_wheelF = self.rotated.get_rect(center=(self.xF, self.yF))

            self.wheelR = pygame.image.load(Wheel_Robot)
            self.rotated_wheelR = self.wheelR
            self.rect_wheelR = self.rotated.get_rect(center=(self.xR, self.yR))

        def setup_control(self):
            # Control variables
            self.vel = 3
            self.acc = 0
            self.theta = 0
            self.delta = 0
            self.alpha = 0
            self.length = 100
            self.kaapa = 0
            self.desired = 0.1
            self.ld = 0

            # path following
            self.error = 0
            self.doterror = 0
            self.sumerror = 0

            self.x1 = self.xpath[0]
            self.y1 = self.ypath[0]
            self.x2 = self.xpath[1]
            self.y2 = self.ypath[1]
            self.x3 = self.xpath[2]
            self.y3 = self.ypath[2]
            self.GoalFlag = False
            self.index = 0

        setup_kinematics(self)
        setup_graphics(self)
        setup_control(self)

        # movement
        self._phi, self._theta = 0, 0
        self._u1, self._u2 = 0, 0
        self._velocity = pygame.math.Vector2(0, 0)
        self._position = pygame.math.Vector2(self.rect.x, self.rect.y)
        self._reversing = False

    def step(self, action):
        """Sprite update function, calcualtes any new position"""
        self.accelerate(action[0])
        self.turn(action[1])

        if action[2] > 0:
            self.brake()

        self._velocity.x = self._u1 * math.cos(self._theta)
        self._velocity.y = self._u1 * math.sin(self._theta)

        self._position += self._velocity
        self.rect.center = (round(self._position[0]), round(self._position[1]))

    def accelerate(self, amount):
        """Increase the speed either forward or reverse"""
        if not self._reversing:
            self._u1 += amount
        else:
            self._u1 -= amount

    def brake(self):
        """Slow the car by half"""
        self._u1 /= 2
        if abs(self._u1) < 0.1:
            self._u1 = 0

    def turn(self, angle_degrees):
        """Adjust the angle the car is heading"""
        #    self._phi = np.math.radians(angle_degrees)
        #        self._theta += (1 / self._lenght) * math.tan(self._phi) * self._u1
        self._theta += math.radians(angle_degrees * 10)

    def reverse(self):
        """Change forward/reverse, reset any speed to 0"""
        self._u1 = 0
        self._reversing = not self._reversing

    def draw(self, display):
        pygame.draw.rect(
            display,
            self.color,
            self._draw_rect,
        )

    @property
    def get_theta(self):
        return self._theta

    @property
    def position(self):
        return self._position

    @property
    def pose(self):
        return pygame.math.Vector3(self._position.x, self._position.y, self.theta)

    # def draw(self, map):
    #    map.blit(self.rotated, self.rect)

    def move(self, dt, event=None):
        self.get_steering()
        self.beta = math.atan(self.lR / (self.lR + self.lF)) * math.tan(self.theta)
        self.psi += (
            self.u / (self.lF + self.lR) * math.cos(self.beta) * math.tan(self.theta)
        ) * dt
        self.x += (self.u * math.cos(self.psi + self.beta)) * dt
        self.y += (self.u * math.sin(self.psi + self.beta)) * dt
        self.xF = self.x + self.lF * math.cos(self.psi)
        self.yF = self.y + self.lF * math.sin(self.psi)
        self.xR = self.x - self.lR * math.cos(self.psi)
        self.yR = self.y - self.lR * math.sin(self.psi)

        self.rotated = pygame.transform.rotozoom(self.img, math.degrees(-self.psi), 1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))
        self.rotated_wheelF = pygame.transform.rotozoom(
            self.wheelF, math.degrees(-(self.psi + self.theta)), 1
        )
        self.rotated_wheelR = pygame.transform.rotozoom(
            self.wheelR, math.degrees(-self.psi), 1
        )
        self.rect_wheelF = self.rotated_wheelF.get_rect(center=(self.xF, self.yF))
        self.rect_wheelR = self.rotated_wheelR.get_rect(center=(self.xR, self.yR))

    def get_steering(self):
        self.get_error()
        P = 0.03
        D = 0.01
        self.theta = P * self.error

    def get_error(self):
        p0 = [self.xF, self.yF]
        p1 = [self.x1, self.y1]
        p2 = [self.x2, self.y2]
        p3 = [self.x3, self.y3]
        error1 = utils.distancepointline(p0, p1, p2)
        error2 = utils.distancepointline(p0, p2, p3)

        if self.x3 != self.xpath[-1]:
            if abs(error1) < abs(error2):
                self.error = error1
            else:
                self.error = error2
                self.index += 1
                self.x1 = self.x2
                self.y1 = self.y2
                self.x2 = self.x3
                self.y2 = self.y3
                self.x3 = self.xpath[self.index + 2]
                self.y3 = self.ypath[self.index + 2]
        else:
            if self.GoalFlag:
                self.error = utils.distancepointline(p0, p2, p3)
            else:
                if abs(error1) < abs(error2):
                    self.error = error1
                else:
                    self.GoalFlag = True
                    self.error = error2
        self.sumerror += self.error
        return self.error

    def carfoundgoal(self):
        dx = self.xpath[-1] - self.x
        dy = self.ypath[-1] - self.y
        dist = (dx**2 + dy**2) ** 0.5
        dgoal = 30
        if dist < dgoal:
            print("found goal")
            print("accumulated error(UoL):", int(self.sumerror))
            return False
        else:
            return True
