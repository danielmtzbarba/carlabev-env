import pygame 

class Renderer:
    def __init__(self, size, fps=60):
        self.window, self.clock = None, None
        self.size, self.fps = size, fps

    def setup(self):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("CarlaBEV — Simulation Viewer")
        self.window = pygame.display.set_mode((self.size, self.size))
        self.clock = pygame.time.Clock()

    def render(self, surface):
        if self.window is None:
            self.setup()
            
        self.window.blit(surface, surface.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.fps)
