import pygame 

class Button:
    def __init__(self, rect, font, text="Button"):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.text = text
        self.pressed = False

        # Colors
        self.default_color = (41, 128, 64)
        self.hover_color   = (49, 148, 75)
        self.pressed_color = (32, 104, 51)
        self.text_color = (255, 255, 255)

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

        pygame.draw.rect(screen, color, self.rect, border_radius=12)
        txt_surf = self.font.render(self.text, True, self.text_color)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)
