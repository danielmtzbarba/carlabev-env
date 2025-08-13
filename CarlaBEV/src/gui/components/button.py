import pygame 

class Button:
    def __init__(self, rect, font, text="Button"):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.text = text
        self.pressed = False

        # Colors
        self.default_color = (50, 150, 50)    # normal
        self.hover_color   = (70, 180, 70)    # hover
        self.pressed_color = (30, 100, 30)    # clicked

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.pressed and self.rect.collidepoint(event.pos):
                self.pressed = False
                return True  # Action fired on release inside button
            self.pressed = False
        return False

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        if self.pressed:
            color = self.pressed_color
        elif self.rect.collidepoint(mouse_pos):
            color = self.hover_color
        else:
            color = self.default_color

        pygame.draw.rect(screen, color, self.rect, border_radius=6)
        txt_surf = self.font.render(self.text, True, (255, 255, 255))
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)
