import pygame 

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
        self.selection = None

    def add_actor(self, category, actor_name):
        if category in self.categories:
            self.categories[category].append(actor_name)
            self.selection = category, len(self.categories[category])-1

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.selection = (cat, idx, self.categories[cat][idx])
            print(self.selection)
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
                if self.selection is not None:
                    if f'{self.selection[0]}-{self.selection[1]}' == item:
                        pygame.draw.rect(screen, self.hover_color, item_rect)

                txt = self.font.render(item, True, self.text_color)
                screen.blit(txt, (item_rect.x + 5, item_rect.y + 2))
                y += 22
