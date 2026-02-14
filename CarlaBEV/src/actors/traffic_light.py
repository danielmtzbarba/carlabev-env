import pygame
from CarlaBEV.src.actors.actor import Actor
from CarlaBEV.src.gui.settings import Settings as cfg

class TrafficLightState:
    RED = 0
    YELLOW = 1
    GREEN = 2

class TrafficLight(Actor):
    def __init__(
        self,
        pos_x,
        pos_y,
        orientation='horizontal', # 'horizontal' or 'vertical' strip
        signal_state=TrafficLightState.RED,
        width=4,
        length=20
    ):
        """
        pos_x, pos_y: Center position of the traffic light strip (scaled coords)
        """
        super().__init__(id="traffic_light", actor_size=1) # size dummy
        
        self.x = pos_x
        self.y = pos_y
        self.orientation = orientation
        self.signal_state = signal_state
        self.width = width
        self.length = length
        
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
        # Static actor, no controller needed
        pass

    def step(self, t=0.0, dt=0.05):
        # Static for now, or could implement timing logic here
        pass

    def draw(self, screen):
        # Override draw to ensure correct positioning with camera offset if needed, 
        # OR assume screen is the map_surface where offsets are already handled or not needed.
        # Looking at Node.render: uses self.draw_x = self._x + cfg.offx
        # So we should apply offset.
        
        draw_x = self.x + cfg.offx
        draw_y = self.y + cfg.offy
        
        if self.orientation == 'horizontal':
            w, h = self.length, self.width
            draw_rect = pygame.Rect(draw_x - w/2, draw_y - h/2, w, h)
        else: # vertical
            w, h = self.width, self.length
            draw_rect = pygame.Rect(draw_x - w/2, draw_y - h/2, w, h)
            
        pygame.draw.rect(screen, self._color, draw_rect)
