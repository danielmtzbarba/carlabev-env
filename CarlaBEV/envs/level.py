from numpy import random
import pygame

from .utils import scale_coords, target_locations, load_map, pedestrian_locations


class Pedestrian(pygame.sprite.Sprite):
    def __init__(self, color=(255, 0, 0), scale=1):
        pygame.sprite.Sprite.__init__(self)
        id = random.randint(0, 4)
        target_location = scale_coords(pedestrian_locations[id], scale)
        x, y = target_location[0], target_location[1]
        self._x0, self._y0 = x, y
        self.position = pygame.math.Vector2(x, y)
        self._behavior = random.randint(0, 2)
        self._scale = scale
        self.size = int(16 / scale)
        self.color = color
        self.rect = pygame.Rect((x, y, self.size, self.size))
        self.dir = 1

        self._running = False
        self._velocity = pygame.math.Vector2(0, 0)

    def reset(self):
        self.rect = pygame.Rect((self._x0, self._y0, self.size, self.size))

    def trigger(self):
        self._running = True

    def draw(self, map):
        if self._running:
            self.move()
            self.rect = pygame.draw.rect(map, self.color, self.rect)

    def move(self):
        if self._behavior == 0:
            if self.position.y > self._y0 + int(150 / self._scale):
                self.dir = -1
            if self.position.y < self._y0:
                self.dir = 1

        else:
            if self.position.y > self._y0 + int(150 / self._scale):
                self.dir = 0

        self._velocity.y = 1
        self._velocity.x = 0
        self.position += self.dir * self._velocity
        self.rect.center = (round(self.position[0]), round(self.position[1]))

    @property
    def pose(self):
        return pygame.math.Vector3(self.position.x, self.position.y, 0)


class Target(pygame.sprite.Sprite):
    lenght, width = 130, 2

    def __init__(self, target_id, color=(0, 255, 0), scale=1):
        pygame.sprite.Sprite.__init__(self)
        target_location = scale_coords(target_locations[target_id], scale)
        x, y = target_location[0], target_location[1]
        self.position = pygame.math.Vector2(x, y)
        self.lenght = int(self.lenght / scale)
        self.color = color
        self.rect = (
            int(x - self.width / 2),
            int(y - self.lenght / 2),
            self.width,
            self.lenght,
        )

    def draw(self, map):
        self.rect = pygame.draw.rect(map, self.color, self.rect)

    @property
    def pose(self):
        return pygame.math.Vector3(self.position.x, self.position.y, 0)


class Scene(object):
    def __init__(self, map_surface, size) -> None:
        self._map_arr, self._map_img = load_map(size)
        self._map = map_surface
        self._curr_goal_id = 0
        self._size = size
        self._scale = int(1024 / size)
        self._const = size / 4
        self.trigger = random.randint(0, 5)
        self._scene_setup(target_id=self._curr_goal_id)

    def _scene_setup(self, target_id):
        self._pedestrian = Pedestrian(scale=self._scale)
        self.next_target(target_id)

    def next_target(self, target_id):
        if target_id == self.trigger:
            self._pedestrian.trigger()
        self._target = Target(target_id, scale=self._scale)

    def draw(self):
        self._map.blit(self._map_img, (0, 0))
        self._target.draw(self._map)
        self._pedestrian.draw(self._map)

    def step(self):
        self.draw()

    def got_target(self, hero):
        offsetx = self._const - hero.rect.w / 2
        offsety = self._const - hero.rect.w / 2
        dummy_rect = pygame.Rect(
            hero.rect.x + offsetx,
            hero.rect.y + offsety,
            hero.rect.w + 1,
            hero.rect.w + 1,
        )
        result = dummy_rect.colliderect(self._target.rect)
        return result

    def hit_pedestrian(self, hero):
        offsetx = self._const - hero.rect.w / 2
        offsety = self._const - hero.rect.w / 2
        dummy_rect = pygame.Rect(
            hero.rect.x + offsetx, hero.rect.y + offsety, hero.rect.w, hero.rect.w
        )
        result = dummy_rect.colliderect(self._pedestrian.rect)
        return result

    @property
    def target_position(self):
        return self._target.position

    @property
    def target_pose(self):
        return self._target.pose
