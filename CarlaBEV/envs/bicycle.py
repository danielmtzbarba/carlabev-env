import math
import pygame
import os

Projectfolder_image = "assets/"

Body_Robot = os.path.join(Projectfolder_image, "Body.png")
Wheel_Robot = os.path.join(Projectfolder_image, "wheel.png")


class Robot(pygame.sprite.Sprite):
    m2p = 3779.52  # meters 2 pixels

    def __init__(self, startpos, path, car_size=32, color=(0, 7, 175)):
        pygame.sprite.Sprite.__init__(self)
        # my stuff
        size = 128
        self.scale = int((1024 / size))
        self._w = int(car_size / self.scale)
        self._l = 2 * self._w
        self._spawn_location = (
            int(startpos[0] - self._l / 2),
            int(startpos[1] - self._w / 2),
        )
        self.color = color
        #
        self.xpath, self.ypath = path
        self.x, self.y = startpos
        #
        self.beta = 0
        self.r = 0
        self.u = 180  # pix/sec
        self.x_back = 0
        self.y_back = 0
        self.maxspeed = 0.02 * self.m2p
        self.minspeed = -0.02 * self.m2p
        # graphics
        self.img = pygame.image.load(Body_Robot)
        # Scale the image to your needed size
        self.img = pygame.transform.scale(self.img, (self._l, self._w))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

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

        self.wheelF = pygame.image.load(Wheel_Robot)
        self.rotated_wheelF = self.wheelF
        self.rect_wheelF = self.rotated.get_rect(center=(self.xF, self.yF))

        self.wheelR = pygame.image.load(Wheel_Robot)
        self.rotated_wheelR = self.wheelR
        self.rect_wheelR = self.rotated.get_rect(center=(self.xR, self.yR))

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

    def draw(self, map):
        map.blit(self.rotated, self.rect)

    def dist(self, point1, point2):
        (x1, y1) = point1
        (x2, y2) = point2
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        px = (x1 - x2) ** (2)
        py = (y1 - y2) ** (2)
        distance = (px + py) ** (0.5)
        return distance

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
        error1 = self.distancepointline(p0, p1, p2)
        error2 = self.distancepointline(p0, p2, p3)

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
                self.error = self.distancepointline(p0, p2, p3)
            else:
                if abs(error1) < abs(error2):
                    self.error = error1
                else:
                    self.GoalFlag = True
                    self.error = error2
        self.sumerror += self.error
        return self.error

    def get_current_path(self):
        x3 = self.x1

    def distancepointline(self, p0, p1, p2):
        (x0, y0) = p0
        (x1, y1) = p1
        (x2, y2) = p2
        px = (float(x1) - float(x2)) ** 2
        py = (float(y1) - float(y2)) ** 2
        num = (x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)
        den = (px + py) ** (0.5)
        return num / den

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
