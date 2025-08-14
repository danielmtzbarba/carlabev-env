import pygame
import numpy as np

from CarlaBEV.src.gui.components import Button, Selector, TextBox, ListBox
from CarlaBEV.src.gui.settings import Settings as cfg


class GUI():
    def __init__(self):
        pygame.display.set_caption("Traffic Scenario Designer")
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        self.font = pygame.font.SysFont(None, 24)

        # Simulation state for timeline
        self.current_frame = 0
        self.total_frames = 200  # Example: can be updated dynamically

        # Elements
        self.scene_name = TextBox((cfg.margin_x, 50, 180, 30), self.font, "Scene1")
        self.actor_selector = Selector((cfg.margin_x, 100, 180, 100), self.font,
                                       ["Agent", "Vehicle", "Pedestrian"])
       # self.lane_selector = Selector((150, 100, 180, 100), self.font,
       #                                ["L", "C", "R"])
        self.add_actor_btn = Button((cfg.margin_x, 200, 180, 30), self.font, "Add Actor")
        self.add_rdm_actor_btn = Button((cfg.margin_x, 240, 180, 30), self.font, "Random Actor")
        self.del_btn = Button((cfg.margin_x, 280, 180, 30), self.font, "Delete Actor")
        self.listbox = ListBox((cfg.margin_x, 320, 180, 300), self.font)
        self.save_btn = Button((cfg.margin_x, 640, 180, 30), self.font, "Save scene")
        self.play_btn = Button((cfg.margin_x + 1050, 300, 180, 30), self.font, "Play Scene")
        # FOV display rect
        self.fov_rect = pygame.Rect(self.screen.get_width() - 220, 20, 200, 200)
        self.timeline_rect = pygame.Rect(cfg.left_panel_w + 20, self.screen.get_height() - 40, 
                                         self.screen.get_width() - 2 * cfg.left_panel_w - 40, 20)

    def handle_timeline_event(self, event):
        """Updates frame based on user dragging the handle."""
        if event.type == pygame.MOUSEBUTTONDOWN and self.timeline_rect.collidepoint(event.pos):
            self.update_frame_from_mouse(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
            if self.timeline_rect.collidepoint(event.pos):
                self.update_frame_from_mouse(event.pos[0])

    def handle_event(self, event):
        self.scene_name.handle_event(event)

        if not self.add_mode:
            self.actor_selector.handle_event(event)
            #self.lane_selector.handle_event(event)
            self.handle_timeline_event(event)
        
        # Toggle Add mode
        if self.add_actor_btn.handle_event(event):
            # Add actor
            if not self.add_mode:
                self.toggle_add_mode()
                return "add_actor"
        
        if event.type == pygame.MOUSEBUTTONUP and self.add_mode: 
            self.add_actor(event)
        
        if self.add_rdm_actor_btn.handle_event(event):
            self.add_rdm_actor()

        if not self.play_mode and self.play_btn.handle_event(event):
            self.play_scene()
            return True

        if not self.add_mode and self.save_btn.handle_event(event):
            self.save_scene(self.scene_name.text)

        return None

    def draw_fov(self, vehicle_surface=None):
        """Draws the cropped FOV view."""
        pygame.draw.rect(self.screen, (0,0,0), self.fov_rect, 2)
        if vehicle_surface is not None:
            vehicle_surface = np_to_surface(vehicle_surface)
            scaled = pygame.transform.scale(vehicle_surface, (self.fov_rect.width, self.fov_rect.height))
            self.screen.blit(scaled, self.fov_rect.topleft)
        else:
            text = self.font.render("No FOV", True, (100,100,100))
            self.screen.blit(text, (self.fov_rect.x+50, self.fov_rect.y+90))

    def draw_timeline(self):
        """Draw timeline bar and handle."""
        pygame.draw.rect(self.screen, (180,180,180), self.timeline_rect)
        # Handle position
        handle_x = self.timeline_rect.x + int((self.current_frame/self.total_frames) * self.timeline_rect.width)
        pygame.draw.circle(self.screen, (0,120,215), (handle_x, self.timeline_rect.centery), 8)

    def draw_gui(self):
        # Draw background panel
        pygame.draw.rect(self.screen, cfg.grey, (0, 0, cfg.left_panel_w, self.screen.get_height()))
        pygame.draw.line(self.screen, cfg.black, (cfg.left_panel_w, 0), (cfg.left_panel_w, self.screen.get_height()), 2)

        # Draw background panel
        pygame.draw.rect(self.screen, cfg.grey, (1025, 0, cfg.right_panel_w, self.screen.get_height()))
        pygame.draw.line(self.screen, cfg.black, (1025, 0), (1025, self.screen.get_height()), 2)

        # Title
        title = self.font.render("Scenario Designer", True, cfg.black)
        self.screen.blit(title, (cfg.margin_x, 15))

        # Elements
        self.scene_name.draw(self.screen)
        self.actor_selector.draw(self.screen)
#        self.lane_selector.draw(self.screen)
        self.add_actor_btn.draw(self.screen)
        self.add_rdm_actor_btn.draw(self.screen)
        self.listbox.draw(self.screen)
        self.del_btn.draw(self.screen)
        self.save_btn.draw(self.screen)
        self.play_btn.draw(self.screen)

        # NEW: draw FOV
        self.draw_fov(None)  # Pass actual cropped vehicle surface here

        # NEW: draw timeline
        self.draw_timeline()

    def update_frame_from_mouse(self, mouse_x):
        rel_x = max(0, min(mouse_x - self.timeline_rect.x, self.timeline_rect.width))
        self.current_frame = int((rel_x / self.timeline_rect.width) * self.total_frames)


def np_to_surface(arr):
    return pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
