import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
BUTTON_COLOR = (50, 150, 50)
BUTTON_HOVER = (70, 180, 70)
BLUE = (0, 120, 215)


class TextBox:
    def __init__(self, rect, font, text=""):
        self.rect = pygame.Rect(rect)
        self.color = WHITE
        self.text = text
        self.font = font
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, 0)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        txt_surface = self.font.render(self.text, True, BLACK)
        screen.blit(txt_surface, (self.rect.x + 5, self.rect.y + 5))


class Selector:
    def __init__(self, rect, font, options, selected=0):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.options = options
        self.selected = selected
        self.option_rects = []
        self._build_option_rects()

    def _build_option_rects(self):
        self.option_rects = []
        y = self.rect.y
        for i, opt in enumerate(self.options):
            r = pygame.Rect(self.rect.x + 10, y + i * 30, 20, 20)
            self.option_rects.append(r)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, r in enumerate(self.option_rects):
                if r.collidepoint(event.pos):
                    self.selected = i

    def draw(self, screen):
        for i, r in enumerate(self.option_rects):
            pygame.draw.rect(screen, BLACK, r, 2)
            if i == self.selected:
                pygame.draw.circle(screen, BLUE, r.center, 7)
            text_surf = self.font.render(self.options[i], True, BLACK)
            screen.blit(text_surf, (r.right + 5, r.y - 2))
    
    @property
    def selection(self):
        return self.options[self.selected]


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

class ListBox:
    def __init__(self, rect, font):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.categories = {"Agent": [], "Vehicle": [], "Pedestrian": []}
        self.hovered = None  # (category, index)

        # Colors
        self.bg_color = (240, 240, 240)
        self.border_color = (0, 0, 0)
        self.text_color = (0, 0, 0)
        self.header_color = (180, 180, 180)
        self.hover_color = (200, 220, 255)

    def add_actor(self, category, actor_name):
        if category in self.categories:
            self.categories[category].append(actor_name)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered is not None:
                cat, idx = self.hovered
                return (cat, idx, self.categories[cat][idx])  # category, index, name
        return None

    def draw(self, screen):
        # Panel background
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, 2)

        y = self.rect.y + 5
        mouse_pos = pygame.mouse.get_pos()
        self.hovered = None

        # Draw categories & items
        for cat, items in self.categories.items():
            # Draw header
            header = self.font.render(cat, True, self.text_color)
            pygame.draw.rect(screen, self.header_color,
                             (self.rect.x + 2, y, self.rect.width - 4, 20))
            screen.blit(header, (self.rect.x + 8, y + 2))
            y += 24

            # Draw items
            for i, item in enumerate(items):
                item_rect = pygame.Rect(self.rect.x + 5, y, self.rect.width - 10, 20)

                # Hover effect
                if item_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, self.hover_color, item_rect)
                    self.hovered = (cat, i)

                txt = self.font.render(item, True, self.text_color)
                screen.blit(txt, (item_rect.x + 5, item_rect.y + 2))
                y += 22

class GUI:
    def __init__(self):
        self.font = pygame.font.SysFont(None, 24)

        # Left panel width
        self.panel_width = 200

        # Elements
        self.scene_name = TextBox((10, 50, 180, 30), self.font, "Scene1")
        self.actor_selector = Selector((10, 100, 180, 100), self.font,
                                       ["Agent", "Vehicle", "Pedestrian"])
        self.lane_selector = Selector((130, 100, 180, 100), self.font,
                                       ["L", "C", "R"])
        self.add_actor_btn = Button((10, 200, 180, 30), self.font, "Add Actor")
        self.listbox = ListBox((10, 260, 180, 300), self.font)

        self.save_btn = Button((10, 600, 180, 30), self.font, "Save scene")


    def handle_event(self, event):
        self.scene_name.handle_event(event)

        if not self.add_mode:
            self.actor_selector.handle_event(event)
            self.lane_selector.handle_event(event)
        
        # Toggle Add mode
        if self.add_actor_btn.handle_event(event):
            # Add actor
            if not self.add_mode:
                self.toggle_add_mode()
                return "add_actor"
            
        if event.type == pygame.MOUSEBUTTONUP and self.add_mode: 
            self.add_actor(event)
        
        if not self.add_mode and self.save_btn.handle_event(event):
            self.save_scene(self.scene_name.text)
        return None

    def draw_gui(self):
        # Draw background panel
        pygame.draw.rect(self.screen, GREY, (0, 0, self.panel_width, self.screen.get_height()))
        pygame.draw.line(self.screen, BLACK, (self.panel_width, 0), (self.panel_width, self.screen.get_height()), 2)

        # Title
        title = self.font.render("Scenario Designer", True, BLACK)
        self.screen.blit(title, (10, 15))

        # Elements
        self.scene_name.draw(self.screen)
        self.actor_selector.draw(self.screen)
        self.lane_selector.draw(self.screen)
        self.add_actor_btn.draw(self.screen)
        self.listbox.draw(self.screen)
        self.save_btn.draw(self.screen)
