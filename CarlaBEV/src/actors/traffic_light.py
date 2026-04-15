import pygame
from CarlaBEV.src.actors.actor import Actor

class TrafficLightState:
    RED = 0
    YELLOW = 1
    GREEN = 2

class TrafficLight(Actor):
    def __init__(
        self,
        pos_x,
        pos_y,
        map_size=128,
        orientation='horizontal', # 'horizontal' or 'vertical' strip
        signal_state=TrafficLightState.RED,
        width=None,
        length=None
    ):
        """
        pos_x, pos_y: Center position of the traffic light strip (scaled coords)
        """
        super().__init__(id="traffic_light", actor_size=1)

        scale = max(1, int(1024 / map_size))
        self.x = pos_x
        self.y = pos_y
        self._map_size = map_size
        self.orientation = orientation
        self.signal_state = signal_state
        self.width = width if width is not None else max(2, int(16 / scale))
        self.length = length if length is not None else max(6, int(48 / scale))

        self._set_color()
        self._update_rect()

    def _set_color(self):
        if self.signal_state == TrafficLightState.RED:
            self._color = (255, 0, 0)
        elif self.signal_state == TrafficLightState.YELLOW:
            self._color = (255, 255, 0)
        elif self.signal_state == TrafficLightState.GREEN:
            self._color = (0, 255, 0)
        else:
            self._color = (100, 100, 100)
    
    # ... (skipping _update_rect as it doesn't use state)

    def _update_rect(self):
        # Center the rect around x, y
        if self.orientation == 'horizontal':
            w, h = self.length, self.width
        else:
            w, h = self.width, self.length
            
        left = self.x - w / 2
        top = self.y - h / 2
        
        # We store rect for drawing. 
        self.rect = pygame.Rect(left, top, w, h)

    def set_signal_state(self, new_state):
        self.signal_state = new_state
        self._set_color()

    def reset(self):
        self._update_rect()

    def step(self, t=0.0, dt=0.05):
        return

    def draw(self, screen, frame=None):
        self._update_rect()
        if self.orientation == 'horizontal':
            w, h = self.length, self.width
            draw_rect = pygame.Rect(self.x - w / 2, self.y - h / 2, w, h)
        else:
            w, h = self.width, self.length
            draw_rect = pygame.Rect(self.x - w / 2, self.y - h / 2, w, h)

        pygame.draw.rect(screen, self._color, draw_rect)

    def isCollided(self, hero, offset):
        return self.id, False, 9999
