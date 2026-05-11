import math
import logging
import os
import pygame
import numpy as np

from CarlaBEV.src.gui.components import Button, Selector, TextBox, ListBox
from CarlaBEV.src.gui.settings import Settings
from CarlaBEV.src.scenes.scenarios.specs import get_scenario_spec, list_scenario_ids

logger = logging.getLogger("carlabev.scene_designer")

try:
    import pygame._sdl2.video as sdl2_video
except Exception:  # pragma: no cover - optional SDL2 path
    sdl2_video = None


class GUI:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        pygame.display.set_caption("Traffic Scenario Designer")
        self.min_window_size = (
            960,
            640,
        )
        self._desktop_size = self._get_desktop_size()
        self._last_desktop_size = self._desktop_size
        self._display_index = self._get_display_index()
        self._last_display_index = self._display_index
        self._user_resized_window = False
        self._last_display_snapshot = None
        self._resolved_layout_preset = self._resolve_layout_preset(self._desktop_size)
        self.layout_cfg = self._get_runtime_layout_config(self._resolved_layout_preset)
        self.min_window_size = (
            self.layout_cfg.min_window_width,
            self.layout_cfg.min_window_height,
        )
        window_width, window_height = self._compute_window_size(self._desktop_size)
        self.screen = pygame.display.set_mode(
            (window_width, window_height),
            pygame.RESIZABLE,
        )
        actual_window_size = self.screen.get_size()
        self._refresh_layout_preset(self._desktop_size, actual_window_size, "startup")
        self._apply_responsive_metrics(*actual_window_size, self._desktop_size)

        self.colors = {
            "app_bg": (232, 235, 239),
            "panel_bg": (244, 246, 248),
            "card_bg": (255, 255, 255),
            "canvas_bg": (210, 214, 219),
            "canvas_frame": (185, 190, 197),
            "border": (182, 188, 196),
            "text": (28, 31, 36),
            "muted": (96, 104, 117),
            "accent": (29, 110, 235),
            "accent_soft": (227, 238, 255),
            "success": (41, 128, 64),
            "success_soft": (229, 242, 232),
        }

        self.left_panel_rect = pygame.Rect(0, 0, 320, self.screen.get_height())
        self.right_panel_rect = pygame.Rect(self.screen.get_width() - 260, 0, 260, self.screen.get_height())
        self.left_body_rect = pygame.Rect(0, 0, 0, 0)
        self.map_rect = pygame.Rect(0, 0, 100, 100)
        self._map_surface_size = (128, 128)
        self._map_crop_rect = pygame.Rect(0, 0, 128, 128)
        self._layout_ready = False
        self.left_scroll_offset = 0
        self.left_content_height = 0

        self.current_frame = 0
        self.total_frames = 200
        self.status_lines = []
        self._last_window_size = actual_window_size
        self.map_click_mode = None
        self.pending_actor_type = None
        self.pending_actor_start = None
        self.designed_actors = []
        self.selected_actor_index = None
        self.actor_row_rects = []
        self.editor_tab = "parameters"
        self.editor_tab_rects = {}

        self.scene_name = TextBox((0, 0, 100, 36), self.font, "Scene1")
        self.scenario_selector = Selector((0, 0, 100, 100), self.font, list_scenario_ids())
        self.level_selector = Selector((0, 0, 100, 100), self.font, ["Level 1", "Level 2", "Level 3", "Level 4"])
        self.field_boxes = {}
        self.active_fields = []
        self.sync_scenario_form()

        self.anchor_btn = Button((0, 0, 120, 34), self.font, "Anchor On Map")
        self.del_btn = Button((0, 0, 120, 34), self.font, "Delete Actor")
        self.place_ego_btn = Button((0, 0, 120, 34), self.font, "Place Ego")
        self.place_vehicle_btn = Button((0, 0, 120, 34), self.font, "Place Vehicle")
        self.place_ped_btn = Button((0, 0, 120, 34), self.font, "Place Ped")
        self.listbox = ListBox((0, 0, 100, 100), self.font)
        self.save_btn = Button((0, 0, 120, 34), self.font, "Save Config")
        self.load_btn = Button((0, 0, 120, 34), self.font, "Load Config")
        self.add_rdm_actor_btn = Button((0, 0, 120, 34), self.font, "Random Scene")
        self.play_btn = Button((0, 0, 120, 34), self.font, "Play Preview")

        self.fov_rect = pygame.Rect(0, 0, 180, 180)
        self.timeline_rect = pygame.Rect(0, 0, 100, 14)
        self.section_rects = {}
        self._log_display_context("startup")

    def status_summary_lines(self):
        return [
            f"Scenario: {self.scenario_spec.display_name}",
            f"Level: {self.level_selector.selection}",
            f"Actors: {len(self.designed_actors)}",
        ]

    def _resolve_layout_preset(self, desktop_size, window_size=None):
        requested = self.settings.designer_layout_preset
        if requested != "auto":
            return requested
        desktop_w, desktop_h = desktop_size
        window_w, window_h = window_size or desktop_size
        effective_h = min(desktop_h, window_h)
        effective_w = min(desktop_w, window_w)
        if desktop_w >= 1800 and desktop_h >= 1000 and window_w >= 1400 and window_h >= 850:
            return "wide"
        if effective_h <= 620 and effective_w <= 760:
            return "dense"
        if effective_h <= 760 or effective_w <= 1180:
            return "compact"
        return "comfortable"

    def _get_runtime_layout_config(self, preset_name):
        from CarlaBEV.src.gui.settings import get_designer_layout_config

        return get_designer_layout_config(
            preset_name,
            self.settings.designer_layout_overrides,
        )

    def _get_desktop_size(self):
        desktop_sizes = pygame.display.get_desktop_sizes()
        display_index = self._get_display_index()
        if desktop_sizes:
            if 0 <= display_index < len(desktop_sizes):
                width, height = desktop_sizes[display_index]
            else:
                width, height = desktop_sizes[0]
            return max(width, self.min_window_size[0]), max(height, self.min_window_size[1])
        display = pygame.display.Info()
        return (
            max(display.current_w, self.min_window_size[0]),
            max(display.current_h, self.min_window_size[1]),
        )

    def _get_display_index(self):
        if sdl2_video is not None and pygame.display.get_surface() is not None:
            try:
                return int(sdl2_video.Window.from_display_module().display_index)
            except Exception:
                pass
        return int(os.environ.get("PYGAME_DISPLAY", "0"))

    def _display_snapshot(self):
        info = pygame.display.Info()
        window_size = pygame.display.get_window_size() if pygame.display.get_surface() else None
        return {
            "driver": pygame.display.get_driver(),
            "display_index": self._get_display_index(),
            "display_info": {"w": info.current_w, "h": info.current_h},
            "window_size": window_size,
            "desktop_sizes": pygame.display.get_desktop_sizes(),
            "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
            "display_env": os.environ.get("DISPLAY"),
            "session_type": os.environ.get("XDG_SESSION_TYPE"),
            "layout_preset": self.settings.designer_layout_preset,
            "resolved_layout_preset": self._resolved_layout_preset,
            "layout_overrides": dict(self.settings.designer_layout_overrides),
            "raw_scale": round(getattr(self, "raw_ui_scale", 0.0), 4),
            "ui_scale": round(getattr(self, "ui_scale", 0.0), 4),
            "user_resized_window": self._user_resized_window,
        }

    def _log_display_context(self, reason):
        snapshot = self._display_snapshot()
        logger.info("designer_display_context[%s] %s", reason, snapshot)
        self._last_display_snapshot = snapshot

    def _log_display_changes(self, reason):
        snapshot = self._display_snapshot()
        if self._last_display_snapshot is None:
            self._log_display_context(reason)
            return
        changed = {
            key: {"old": self._last_display_snapshot.get(key), "new": snapshot.get(key)}
            for key in snapshot
            if self._last_display_snapshot.get(key) != snapshot.get(key)
        }
        if changed:
            logger.info("designer_display_change[%s] %s", reason, changed)
            self._last_display_snapshot = snapshot

    def _refresh_layout_preset(self, desktop_size, window_size, reason):
        resolved = self._resolve_layout_preset(desktop_size, window_size)
        if resolved != self._resolved_layout_preset:
            old_preset = self._resolved_layout_preset
            self._resolved_layout_preset = resolved
            self.layout_cfg = self._get_runtime_layout_config(resolved)
            self.min_window_size = (
                self.layout_cfg.min_window_width,
                self.layout_cfg.min_window_height,
            )
            logger.info(
                "designer_layout_preset_change[%s] %s",
                reason,
                {
                    "old": old_preset,
                    "new": resolved,
                    "desktop_size": desktop_size,
                    "window_size": window_size,
                },
            )

    def _compute_window_size(self, desktop_size):
        screen_w = max(desktop_size[0], self.min_window_size[0])
        screen_h = max(desktop_size[1], self.min_window_size[1])
        window_width = int(screen_w * self.layout_cfg.window_width_ratio)
        window_height = int(screen_h * self.layout_cfg.window_height_ratio)
        window_width = max(
            self.min_window_size[0],
            min(window_width, screen_w - self.layout_cfg.window_margin_x),
        )
        window_height = max(
            self.min_window_size[1],
            min(window_height, screen_h - self.layout_cfg.window_margin_y),
        )
        return window_width, window_height

    def _refresh_window_metrics(self):
        surface = pygame.display.get_surface()
        if surface is None:
            return
        current_display_index = self._get_display_index()
        current_desktop_size = self._get_desktop_size()
        current_size = surface.get_size()
        desktop_changed = current_desktop_size != self._last_desktop_size
        display_changed = current_display_index != self._last_display_index
        if display_changed:
            self._display_index = current_display_index
            self._last_display_index = current_display_index
        if desktop_changed:
            self._desktop_size = current_desktop_size
            self._last_desktop_size = current_desktop_size
            if not self._user_resized_window:
                target_size = self._compute_window_size(current_desktop_size)
                if target_size != current_size:
                    self.screen = pygame.display.set_mode(target_size, pygame.RESIZABLE)
                    surface = pygame.display.get_surface()
                    current_size = surface.get_size()
        self._refresh_layout_preset(current_desktop_size, current_size, "refresh")
        if current_size != self._last_window_size or desktop_changed or display_changed:
            self.screen = surface
            self._last_window_size = current_size
            self._apply_responsive_metrics(*current_size, desktop_size=current_desktop_size)
            self.update_layout(self._map_surface_size)
            self._log_display_changes("refresh")

    def _apply_responsive_metrics(self, screen_w, screen_h, desktop_size=None):
        raw_scale = min(
            screen_w / self.layout_cfg.scale_ref_width,
            screen_h / self.layout_cfg.scale_ref_height,
        )
        self.raw_ui_scale = raw_scale
        if raw_scale >= 1.0:
            scale = 1.0 + (raw_scale - 1.0) * 0.45
        else:
            scale = 1.0 - (1.0 - raw_scale) * 0.80
        self.ui_scale = max(self.layout_cfg.scale_min, min(scale, self.layout_cfg.scale_max))

        self.spacing_xs = max(4, int(round(self.layout_cfg.spacing_xs_base * self.ui_scale)))
        self.spacing_sm = max(8, int(round(self.layout_cfg.spacing_sm_base * self.ui_scale)))
        self.spacing_md = max(12, int(round(self.layout_cfg.spacing_md_base * self.ui_scale)))
        self.spacing_lg = max(16, int(round(self.layout_cfg.spacing_lg_base * self.ui_scale)))
        self.panel_padding = max(18, int(round(self.layout_cfg.panel_padding_base * self.ui_scale)))
        self.section_pad = max(10, int(round(self.layout_cfg.section_padding_base * self.ui_scale)))
        self.option_height = max(28, int(round(self.layout_cfg.option_height_base * self.ui_scale)))
        self.option_gap = max(4, int(round(self.layout_cfg.option_gap_base * self.ui_scale)))
        self.textbox_height = max(36, int(round(self.layout_cfg.textbox_height_base * self.ui_scale)))
        self.button_height = max(36, int(round(self.layout_cfg.button_height_base * self.ui_scale)))
        self.timeline_height = max(10, int(round(self.layout_cfg.timeline_height_base * self.ui_scale)))
        self.section_radius = max(14, int(round(self.layout_cfg.section_radius_base * self.ui_scale)))
        self.card_radius = max(14, int(round(self.layout_cfg.card_radius_base * self.ui_scale)))
        self.selector_title_gap = max(14, int(round(self.layout_cfg.section_title_gap_base * self.ui_scale)))
        self.param_label_gap = max(12, int(round(self.layout_cfg.param_label_gap_base * self.ui_scale)))
        self.frame_pad = max(16, int(round(self.layout_cfg.frame_padding_base * self.ui_scale)))
        self.header_x = self.panel_padding
        self.header_y = self.panel_padding
        self.header_gap = max(10, int(round(self.layout_cfg.header_gap_base * self.ui_scale)))
        self.timeline_handle_radius = max(6, int(round(self.layout_cfg.timeline_handle_radius_base * self.ui_scale)))
        self.button_radius = max(10, int(round(self.layout_cfg.button_radius_base * self.ui_scale)))

        self.font = pygame.font.SysFont(
            None,
            max(self.layout_cfg.font_body_min, int(round(self.layout_cfg.font_body_base * self.ui_scale))),
        )
        self.font_small = pygame.font.SysFont(
            None,
            max(self.layout_cfg.font_small_min, int(round(self.layout_cfg.font_small_base * self.ui_scale))),
        )
        self.font_title = pygame.font.SysFont(
            None,
            max(self.layout_cfg.font_title_min, int(round(self.layout_cfg.font_title_base * self.ui_scale))),
        )
        self.font_section = pygame.font.SysFont(
            None,
            max(self.layout_cfg.font_section_min, int(round(self.layout_cfg.font_section_base * self.ui_scale))),
        )
        self.font_button = pygame.font.SysFont(
            None,
            max(self.layout_cfg.font_body_min + 2, int(round((self.layout_cfg.font_body_base + 2) * self.ui_scale))),
        )

        for widget in (
            "save_btn",
            "load_btn",
            "anchor_btn",
            "add_rdm_actor_btn",
            "play_btn",
            "del_btn",
            "place_ego_btn",
            "place_vehicle_btn",
            "place_ped_btn",
        ):
            if hasattr(self, widget):
                getattr(self, widget).font = self.font_button
        if hasattr(self, "scene_name"):
            self.scene_name.font = self.font
        if hasattr(self, "listbox"):
            self.listbox.font = self.font_small
        for widget in ("scenario_selector", "level_selector"):
            if hasattr(self, widget):
                selector = getattr(self, widget)
                selector.font = self.font
                selector.set_metrics(self.option_height, self.option_gap)
        if hasattr(self, "field_boxes"):
            for box in self.field_boxes.values():
                box.font = self.font
        if hasattr(self, "refresh_actor_detail_fonts"):
            self.refresh_actor_detail_fonts()

    def handle_timeline_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.timeline_rect.collidepoint(event.pos):
            self.update_frame_from_mouse(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
            if self.timeline_rect.collidepoint(event.pos):
                self.update_frame_from_mouse(event.pos[0])

    def handle_event(self, event):
        if event.type == pygame.VIDEORESIZE:
            surface = pygame.display.get_surface()
            if surface is None:
                return None
            new_w, new_h = surface.get_size()
            self.screen = surface
            self._last_window_size = surface.get_size()
            self._user_resized_window = True
            self._refresh_layout_preset(self._get_desktop_size(), (new_w, new_h), "videoresize")
            self._apply_responsive_metrics(new_w, new_h, desktop_size=self._get_desktop_size())
            self.update_layout(self._map_surface_size)
            self._log_display_changes("videoresize")
            return None

        self.scene_name.handle_event(event)

        prev_scenario = self.scenario_selector.selected
        self.scenario_selector.handle_event(event)
        if self.scenario_selector.selected != prev_scenario:
            self.sync_scenario_form()

        self.level_selector.handle_event(event)
        for box in self.field_boxes.values():
            box.handle_event(event)
        self.handle_timeline_event(event)
        selected_scene = self.listbox.handle_event(event)
        if selected_scene:
            self.scene_name.text = selected_scene

        if event.type == pygame.MOUSEBUTTONUP:
            for idx, row_rect in enumerate(self.actor_row_rects):
                if row_rect.collidepoint(event.pos):
                    self.selected_actor_index = idx
                    if hasattr(self, "on_selected_actor_changed"):
                        self.on_selected_actor_changed()
                    return None
            for tab_name, tab_rect in self.editor_tab_rects.items():
                if tab_rect.collidepoint(event.pos):
                    self.editor_tab = tab_name
                    if hasattr(self, "on_editor_tab_changed"):
                        self.on_editor_tab_changed()
                    return None

        if hasattr(self, "handle_actor_detail_event") and self.handle_actor_detail_event(event):
            return None

        ignored_click = False
        if event.type == pygame.MOUSEBUTTONUP:
            ignored_click = any(
                rect.collidepoint(event.pos)
                for rect in [
                    self.scene_name.rect,
                    self.scenario_selector.rect,
                    self.level_selector.rect,
                    self.listbox.rect,
                    self.save_btn.rect,
                    self.load_btn.rect,
                    self.anchor_btn.rect,
                    self.del_btn.rect,
                    self.place_ego_btn.rect,
                    self.place_vehicle_btn.rect,
                    self.place_ped_btn.rect,
                    self.add_rdm_actor_btn.rect,
                    self.play_btn.rect,
                ]
            ) or any(box.rect.collidepoint(event.pos) for box in self.field_boxes.values()) or any(
                rect.collidepoint(event.pos) for rect in self.actor_row_rects
            )

        if event.type == pygame.MOUSEBUTTONUP and not ignored_click and self.map_rect.collidepoint(event.pos):
            if self.map_click_mode == "anchor":
                self.map_click_mode = None
                return {"action": "anchor", "pos": event.pos}
            if self.map_click_mode == "actor_start":
                self.pending_actor_start = event.pos
                self.map_click_mode = "actor_end"
                self.set_status("Select the actor end position.", "Second click completes the route.")
                return None
            if self.map_click_mode == "actor_end" and self.pending_actor_start is not None:
                start_pos = self.pending_actor_start
                actor_type = self.pending_actor_type
                self.pending_actor_start = None
                self.pending_actor_type = None
                self.map_click_mode = None
                return {
                    "action": "add_actor",
                    "actor_type": actor_type,
                    "start_pos": start_pos,
                    "end_pos": event.pos,
                }

        if self.add_rdm_actor_btn.handle_event(event):
            self.env.reset(options={"scene": "rdm"})
            self.set_status("Loaded a random traffic scene.", "Choose a scenario and click the map to return to authored mode.")
            return "rdm"

        if self.play_btn.handle_event(event):
            if not self.play_mode:
                self.play_btn.text = "Stop Preview"
                self.play_scene()
                return True
            self.play_btn.text = "Play Preview"
            self.toggle_play_mode()

        if self.save_btn.handle_event(event):
            self.save_scene(self.scene_name.text)

        if self.load_btn.handle_event(event):
            try:
                self.load_scene(self.listbox.selected_item or self.scene_name.text)
            except Exception as exc:
                print(f"Failed to load scene: {exc}")
                self.set_status("Failed to load saved scene.", str(exc))

        if self.anchor_btn.handle_event(event):
            self.map_click_mode = "anchor"
            self.pending_actor_start = None
            self.pending_actor_type = None
            self.set_status("Click the map to place the scenario anchor.", "Anchor placement is active.")
            return None

        if self.place_ego_btn.handle_event(event):
            self.map_click_mode = "actor_start"
            self.pending_actor_type = "agent"
            self.pending_actor_start = None
            self.set_status("Click the ego start position.", "Second click sets the ego end position.")
            return None

        if self.place_vehicle_btn.handle_event(event):
            self.map_click_mode = "actor_start"
            self.pending_actor_type = "vehicle"
            self.pending_actor_start = None
            self.set_status("Click the vehicle start position.", "Second click sets the vehicle end position.")
            return None

        if self.place_ped_btn.handle_event(event):
            self.map_click_mode = "actor_start"
            self.pending_actor_type = "pedestrian"
            self.pending_actor_start = None
            self.set_status("Click the pedestrian start position.", "Second click sets the pedestrian end position.")
            return None

        if self.del_btn.handle_event(event):
            if self.selected_actor_index is not None:
                return {"action": "delete_actor", "index": self.selected_actor_index}
            self.set_status("No actor selected.", "Select an actor row before deleting it.")
            return None

        return None

    def sync_scenario_form(self):
        spec = self.scenario_spec
        self.level_selector.update_options(spec.level_options())
        self.active_fields = list(spec.fields)

        new_boxes = {}
        for field in self.active_fields:
            existing = self.field_boxes.get(field.key)
            text = existing.text if existing is not None else field.default_text()
            new_boxes[field.key] = TextBox((0, 0, 100, 34), self.font, text)
        self.field_boxes = new_boxes

        if self._layout_ready:
            self.update_layout(self._map_surface_size)

    def get_form_values(self):
        values = {}
        for field in self.active_fields:
            values[field.key] = field.parse(self.field_boxes[field.key].text)
        return values

    def update_layout(self, map_surface_size):
        self._map_surface_size = map_surface_size
        screen_w = self.screen.get_width()
        screen_h = self.screen.get_height()
        panel_scale = max(0.92, 0.75 + 0.25 * self.ui_scale)

        left_w = min(
            int(self.layout_cfg.left_panel_width_max * panel_scale),
            max(int(self.layout_cfg.left_panel_width_min * panel_scale), screen_w // 3),
        )
        right_w = min(
            int((self.layout_cfg.right_panel_width_max + 28) * panel_scale),
            max(int((self.layout_cfg.right_panel_width_min + 18) * panel_scale), screen_w // 4),
        )
        center_min_w = self.layout_cfg.center_panel_min_width
        if left_w + right_w + center_min_w > screen_w:
            overflow = left_w + right_w + center_min_w - screen_w
            left_w -= min(
                overflow,
                max(0, left_w - int(self.layout_cfg.left_panel_floor * panel_scale)),
            )
            overflow = max(0, left_w + right_w + center_min_w - screen_w)
            right_w -= min(
                overflow,
                max(0, right_w - int(self.layout_cfg.right_panel_floor * panel_scale)),
            )

        self.left_panel_rect = pygame.Rect(0, 0, left_w, screen_h)
        self.right_panel_rect = pygame.Rect(screen_w - right_w, 0, right_w, screen_h)
        header_title_y = self.header_y
        header_subtitle_y = header_title_y + self.font_title.get_height() + self.spacing_xs
        left_header_h = (
            header_subtitle_y
            + self.font_small.get_height()
            + self.spacing_lg
        )
        self.left_header_title_pos = (self.left_panel_rect.x + self.header_x, header_title_y)
        self.left_header_subtitle_pos = (self.left_panel_rect.x + self.header_x, header_subtitle_y)
        self.right_header_title_pos = (self.right_panel_rect.x + self.header_x, header_title_y)
        self.right_header_subtitle_pos = (self.right_panel_rect.x + self.header_x, header_subtitle_y)
        self.left_body_rect = pygame.Rect(
            self.left_panel_rect.x,
            self.left_panel_rect.y + left_header_h,
            self.left_panel_rect.width,
            self.left_panel_rect.height - left_header_h,
        )
        self.right_body_rect = pygame.Rect(
            self.right_panel_rect.x,
            self.right_panel_rect.y + left_header_h,
            self.right_panel_rect.width,
            self.right_panel_rect.height - left_header_h,
        )

        center_x = self.left_panel_rect.right + self.spacing_lg
        center_w = self.right_panel_rect.left - center_x - self.spacing_lg
        center_top = self.spacing_lg + self.spacing_xs
        center_h = screen_h - max(
            self.layout_cfg.center_bottom_reserve_min,
            int(self.layout_cfg.center_bottom_reserve * self.ui_scale),
        )
        crop_w, crop_h = self._map_crop_rect.size
        map_w, map_h = crop_w, crop_h
        scale = min(center_w / map_w, center_h / map_h)
        draw_w = max(1, int(map_w * scale))
        draw_h = max(1, int(map_h * scale))
        map_x = center_x + max(0, (center_w - draw_w) // 2)
        help_h = self.font_small.get_height() + self.spacing_sm
        block_h = draw_h + self.spacing_lg + help_h + self.timeline_height
        block_top = center_top + max(0, (center_h - block_h) // 2) + self.spacing_sm
        map_y = block_top
        self.map_rect = pygame.Rect(map_x, map_y, draw_w, draw_h)

        fov_size = min(
            self.right_panel_rect.width - 2 * self.panel_padding,
            int(self.layout_cfg.preview_fov_max * self.ui_scale),
        )
        self.timeline_rect = pygame.Rect(
            self.map_rect.x,
            self.map_rect.bottom + self.spacing_lg + help_h,
            self.map_rect.width,
            self.timeline_height,
        )

        self.layout_controls()
        self._layout_ready = True

    def layout_controls(self):
        left_x = self.left_panel_rect.x + self.panel_padding
        left_w = self.left_panel_rect.width - 2 * self.panel_padding
        card_title_h = self.font_section.get_height()
        card_top_pad = self.section_pad
        card_bottom_pad = self.section_pad
        card_header_gap = self.spacing_sm
        field_label_gap = self.spacing_xs
        field_row_gap = self.spacing_sm
        label_h = self.font_small.get_height()
        param_textbox_h = max(28, self.textbox_height - 8)
        setup_option_height = max(22, self.option_height - 6)
        setup_option_gap = max(2, self.option_gap - 1)
        self.scenario_selector.set_metrics(setup_option_height, setup_option_gap)
        self.level_selector.set_metrics(setup_option_height, setup_option_gap)
        scenario_h = len(self.scenario_selector.options) * setup_option_height + max(0, len(self.scenario_selector.options) - 1) * setup_option_gap
        level_h = len(self.level_selector.options) * setup_option_height + max(0, len(self.level_selector.options) - 1) * setup_option_gap
        gap = self.spacing_md
        col_count = 2 if len(self.active_fields) > 3 else 1
        field_w = left_w if col_count == 1 else (left_w - gap) // 2
        row_h = label_h + field_label_gap + param_textbox_h + field_row_gap
        actor_row_h = self.font_small.get_height() + self.spacing_sm
        actor_list_rows = 3
        actor_list_h = actor_row_h * actor_list_rows + self.spacing_sm
        actor_button_gap = self.spacing_sm
        actor_button_w = (left_w - actor_button_gap) // 2
        setup_col_gap = self.spacing_sm
        setup_col_w = (left_w - setup_col_gap) // 2
        tab_gap = self.spacing_sm
        tab_h = max(26, self.button_height - 6)
        tab_w = (left_w - tab_gap) // 2
        actor_detail_h = (
            label_h
            + field_label_gap
            + max(28, self.textbox_height - 8)
            + field_row_gap
            + label_h
            + field_label_gap
            + max(28, self.textbox_height - 8)
            + field_row_gap
            + label_h
            + field_label_gap
            + max(28, self.textbox_height - 8)
        )

        rows = math.ceil(len(self.active_fields) / col_count) if self.active_fields else 0
        params_content_h = 0
        if rows > 0:
            params_content_h = rows * (label_h + field_label_gap + param_textbox_h)
            params_content_h += max(0, rows - 1) * field_row_gap
        params_section_h = card_top_pad + card_title_h + card_header_gap + params_content_h + card_bottom_pad
        actors_section_h = (
            card_top_pad
            + tab_h
            + card_header_gap
            + 2 * self.button_height
            + actor_button_gap
            + self.spacing_md
            + actor_list_h
            + self.spacing_md
            + actor_detail_h
            + card_bottom_pad
        )
        status_section_h = (
            card_top_pad
            + card_title_h
            + card_header_gap
            + len(self.status_summary_lines()) * self.font.get_height()
            + max(0, len(self.status_summary_lines()) - 1) * self.spacing_sm
            + card_bottom_pad
        )
        editor_content_h = max(params_section_h, actors_section_h)
        section_heights = {
            "scene_name": card_top_pad + card_title_h + card_header_gap + self.textbox_height + card_bottom_pad,
            "setup": (
                card_top_pad
                + card_title_h
                + card_header_gap
                + label_h
                + field_label_gap
                + max(scenario_h, level_h)
                + card_bottom_pad
            ),
            "editor": card_top_pad + tab_h + card_header_gap + max(params_content_h, actors_section_h - card_top_pad - tab_h - card_header_gap - card_bottom_pad) + card_bottom_pad,
            "status": status_section_h,
        }
        section_order = ["scene_name", "setup", "editor"]
        natural_gap = self.spacing_lg
        minimum_gap = self.spacing_sm
        tight_gap = self.spacing_xs
        maximum_gap = self.spacing_lg + self.spacing_md
        content_top = self.left_body_rect.y + self.spacing_sm
        content_bottom = self.left_body_rect.bottom - self.spacing_sm
        available_h = max(0, content_bottom - content_top)
        status_gap = self.spacing_lg
        status_y = content_bottom - section_heights["status"]
        upper_content_bottom = max(content_top, status_y - status_gap)
        upper_available_h = max(0, upper_content_bottom - content_top)
        total_section_h = sum(section_heights[name] for name in section_order)
        gap_count = max(1, len(section_order) - 1)
        max_fit_gap = max(0, (upper_available_h - total_section_h) // gap_count)
        if total_section_h + natural_gap * gap_count <= upper_available_h:
            distributed_gap = min(maximum_gap, max(natural_gap, max_fit_gap))
        elif total_section_h + minimum_gap * gap_count <= upper_available_h:
            distributed_gap = max(minimum_gap, max_fit_gap)
        else:
            distributed_gap = max(tight_gap, max_fit_gap)

        used_h = total_section_h + distributed_gap * gap_count
        top_inset = min(self.spacing_md, max(0, (upper_available_h - used_h) // 2))
        y = content_top + top_inset
        self.left_scroll_offset = 0

        self.section_rects["scene_name"] = pygame.Rect(
            left_x - self.section_pad,
            y,
            left_w + 2 * self.section_pad,
            section_heights["scene_name"],
        )
        self.scene_name.rect = pygame.Rect(
            left_x,
            self.section_rects["scene_name"].y + card_top_pad + card_title_h + card_header_gap,
            left_w,
            self.textbox_height,
        )
        y += section_heights["scene_name"] + distributed_gap

        self.section_rects["setup"] = pygame.Rect(
            left_x - self.section_pad,
            y,
            left_w + 2 * self.section_pad,
            section_heights["setup"],
        )
        setup_content_y = self.section_rects["setup"].y + card_top_pad + card_title_h + card_header_gap
        scenario_y = setup_content_y + label_h + field_label_gap
        self.scenario_selector.rect = pygame.Rect(
            left_x,
            scenario_y,
            setup_col_w,
            scenario_h,
        )
        self.scenario_selector._build_option_rects()
        level_y = setup_content_y + label_h + field_label_gap
        self.level_selector.rect = pygame.Rect(
            left_x + setup_col_w + setup_col_gap,
            level_y,
            setup_col_w,
            level_h,
        )
        self.level_selector._build_option_rects()
        y += section_heights["setup"] + distributed_gap

        self.section_rects["editor"] = pygame.Rect(
            left_x - self.section_pad,
            y,
            left_w + 2 * self.section_pad,
            section_heights["editor"],
        )
        tab_y = self.section_rects["editor"].y + card_top_pad
        self.editor_tab_rects = {
            "parameters": pygame.Rect(left_x, tab_y, tab_w, tab_h),
            "actors": pygame.Rect(left_x + tab_w + tab_gap, tab_y, tab_w, tab_h),
        }
        editor_content_y = tab_y + tab_h + card_header_gap
        params_y = editor_content_y
        for idx, field in enumerate(self.active_fields):
            col = idx % col_count
            row = idx // col_count
            box_x = left_x + col * (field_w + gap)
            box_y = params_y + row * row_h + label_h + field_label_gap
            self.field_boxes[field.key].rect = pygame.Rect(box_x, box_y, field_w, param_textbox_h)
        actor_buttons_y = editor_content_y
        self.place_ego_btn.rect = pygame.Rect(left_x, actor_buttons_y, actor_button_w, self.button_height)
        self.place_vehicle_btn.rect = pygame.Rect(
            left_x + actor_button_w + actor_button_gap,
            actor_buttons_y,
            actor_button_w,
            self.button_height,
        )
        self.place_ped_btn.rect = pygame.Rect(
            left_x,
            actor_buttons_y + self.button_height + actor_button_gap,
            actor_button_w,
            self.button_height,
        )
        self.del_btn.rect = pygame.Rect(
            left_x + actor_button_w + actor_button_gap,
            actor_buttons_y + self.button_height + actor_button_gap,
            actor_button_w,
            self.button_height,
        )
        actor_list_y = actor_buttons_y + 2 * self.button_height + actor_button_gap + self.spacing_md
        self.actor_row_rects = []
        actor_card_x = self.section_rects["editor"].x + self.section_pad
        actor_card_w = self.section_rects["editor"].width - 2 * self.section_pad
        for idx in range(actor_list_rows):
            self.actor_row_rects.append(
                pygame.Rect(actor_card_x, actor_list_y + idx * actor_row_h, actor_card_w, actor_row_h)
            )
        self.actor_detail_rect = pygame.Rect(
            actor_card_x,
            actor_list_y + actor_list_h + self.spacing_md,
            actor_card_w,
            actor_detail_h,
        )
        if hasattr(self, "layout_actor_detail_widgets"):
            self.layout_actor_detail_widgets(self.actor_detail_rect)
        y += section_heights["editor"] + distributed_gap

        self.section_rects["status"] = pygame.Rect(
            left_x - self.section_pad,
            status_y,
            left_w + 2 * self.section_pad,
            section_heights["status"],
        )

        content_end = self.section_rects["status"].bottom + self.spacing_sm
        self.left_content_height = max(available_h, content_end - content_top)

        right_x = self.right_panel_rect.x + self.panel_padding
        right_w = self.right_panel_rect.width - 2 * self.panel_padding
        fov_size = min(
            right_w,
            int(self.layout_cfg.preview_fov_max * self.ui_scale),
        )
        right_card_title_h = self.font_section.get_height()
        right_card_top_pad = self.section_pad
        right_card_bottom_pad = self.section_pad
        right_card_header_gap = self.spacing_sm
        summary_line_h = self.font.get_height()
        status_line_h = self.font_small.get_height()
        preview_card_y = self.right_body_rect.y + self.spacing_sm
        preview_card_h = right_card_top_pad + right_card_title_h + max(4, self.spacing_xs // 2) + fov_size + max(6, right_card_bottom_pad - self.spacing_xs)
        preview_card_w = min(
            right_w + 2 * self.section_pad,
            fov_size + 2 * (self.section_pad + self.spacing_sm),
        )
        self.preview_card_rect = pygame.Rect(
            self.right_panel_rect.x + (self.right_panel_rect.width - preview_card_w) // 2,
            preview_card_y,
            preview_card_w,
            preview_card_h,
        )
        self.fov_rect = pygame.Rect(
            self.right_panel_rect.x + (self.right_panel_rect.width - fov_size) // 2,
            self.preview_card_rect.y + right_card_top_pad + right_card_title_h + self.spacing_xs,
            fov_size,
            fov_size,
        )
        anchor_y = self.right_body_rect.y + self.spacing_sm
        self.anchor_btn.rect = pygame.Rect(right_x, anchor_y, right_w, self.button_height)
        self.preview_card_rect.y = self.anchor_btn.rect.bottom + self.spacing_lg
        self.fov_rect.y = self.preview_card_rect.y + right_card_top_pad + right_card_title_h + self.spacing_xs
        action_y = self.preview_card_rect.bottom + self.spacing_lg
        self.play_btn.rect = pygame.Rect(right_x, action_y, right_w, self.button_height)
        self.add_rdm_actor_btn.rect = pygame.Rect(right_x, self.play_btn.rect.bottom + self.spacing_md, right_w, self.button_height)

        list_rows = max(1, min(5, len(self.listbox.items)))
        listbox_row_h = max(24, self.option_height - 4)
        listbox_gap = max(4, self.option_gap)
        self.listbox.set_metrics(listbox_row_h, listbox_gap, padding=self.section_pad)
        listbox_h = (
            2 * self.section_pad
            + list_rows * listbox_row_h
            + max(0, list_rows - 1) * listbox_gap
        )
        self.saved_scenes_rect = pygame.Rect(
            right_x - self.section_pad,
            self.add_rdm_actor_btn.rect.bottom + self.spacing_lg,
            right_w + 2 * self.section_pad,
            listbox_h + self.font_section.get_height() + self.spacing_sm + self.section_pad,
        )
        self.listbox.rect = pygame.Rect(
            self.saved_scenes_rect.x + self.section_pad,
            self.saved_scenes_rect.y + self.section_pad + self.font_section.get_height() + self.spacing_sm,
            self.saved_scenes_rect.width - 2 * self.section_pad,
            listbox_h,
        )
        self.listbox._build_item_rects()
        self.save_btn.rect = pygame.Rect(
            right_x,
            self.saved_scenes_rect.bottom + self.spacing_lg,
            right_w,
            self.button_height,
        )
        self.load_btn.rect = pygame.Rect(
            right_x,
            self.save_btn.rect.bottom + self.spacing_md,
            right_w,
            self.button_height,
        )

    def screen_to_map_pos(self, pos):
        if self.map_rect.width <= 0 or self.map_rect.height <= 0:
            return None
        rel_x = (pos[0] - self.map_rect.x) / self.map_rect.width
        rel_y = (pos[1] - self.map_rect.y) / self.map_rect.height
        map_x = int(self._map_crop_rect.x + rel_x * self._map_crop_rect.width)
        map_y = int(self._map_crop_rect.y + rel_y * self._map_crop_rect.height)
        map_x = max(0, min(self._map_surface_size[0] - 1, map_x))
        map_y = max(0, min(self._map_surface_size[1] - 1, map_y))
        return map_x, map_y

    def map_to_screen_pos(self, map_pos):
        if map_pos is None or self.map_rect.width <= 0 or self.map_rect.height <= 0:
            return None
        map_x, map_y = map_pos
        rel_x = (map_x - self._map_crop_rect.x) / max(1, self._map_crop_rect.width)
        rel_y = (map_y - self._map_crop_rect.y) / max(1, self._map_crop_rect.height)
        screen_x = int(self.map_rect.x + rel_x * self.map_rect.width)
        screen_y = int(self.map_rect.y + rel_y * self.map_rect.height)
        return screen_x, screen_y

    def set_status(self, *lines):
        self.status_lines = [line for line in lines if line]

    def current_map_help_text(self):
        if self.map_click_mode == "anchor":
            return "Click on the map to place the scenario anchor."
        if self.map_click_mode == "actor_start":
            actor_name = (self.pending_actor_type or "actor").replace("agent", "ego")
            return f"Click on the map to place the {actor_name} start point."
        if self.map_click_mode == "actor_end":
            actor_name = (self.pending_actor_type or "actor").replace("agent", "ego")
            return f"Click on the map to place the {actor_name} end point."
        if self.editor_tab == "actors":
            return "Use the actor tools to place start and end points on the map."
        return "Use Anchor On Map or an actor placement button to start editing on the map."

    def draw_fov(self):
        self._draw_card_rect(self.preview_card_rect)
        title = self.font_section.render("FOV Preview", True, self.colors["text"])
        title_rect = title.get_rect(centerx=self.fov_rect.centerx, y=self.preview_card_rect.y + max(6, self.section_pad - self.spacing_xs))
        self.screen.blit(title, title_rect)

        vehicle_surface = self.env.map.canvas
        if vehicle_surface is not None:
            scaled = pygame.transform.scale(vehicle_surface, (self.fov_rect.width, self.fov_rect.height))
            self.screen.blit(scaled, self.fov_rect.topleft)
        else:
            pygame.draw.rect(self.screen, (18, 20, 24), self.fov_rect, border_radius=self.button_radius)
            text = self.font.render("No FOV", True, (220, 223, 228))
            self.screen.blit(text, text.get_rect(center=self.fov_rect.center))

    def draw_timeline(self):
        pygame.draw.rect(self.screen, (208, 214, 220), self.timeline_rect, border_radius=self.timeline_handle_radius)
        if self.total_frames <= 0:
            return
        handle_x = self.timeline_rect.x + int((self.current_frame / self.total_frames) * self.timeline_rect.width)
        handle_x = max(
            self.timeline_rect.x + self.timeline_handle_radius,
            min(self.timeline_rect.right - self.timeline_handle_radius, handle_x),
        )
        pygame.draw.circle(
            self.screen,
            self.colors["accent"],
            (handle_x, self.timeline_rect.centery),
            self.timeline_handle_radius,
        )

    def draw_gui(self):
        self._refresh_window_metrics()
        map_surface = self.env.map.map_surface
        self._map_crop_rect = self._compute_map_crop_rect(map_surface)
        self.update_layout(map_surface.get_size())
        self.screen.fill(self.colors["app_bg"])

        self._draw_side_panels()
        self._draw_left_panel()
        self._draw_center_canvas()
        self.draw_fov()
        self._draw_right_panel()

    def _draw_side_panels(self):
        pygame.draw.rect(self.screen, self.colors["panel_bg"], self.left_panel_rect)
        pygame.draw.rect(self.screen, self.colors["panel_bg"], self.right_panel_rect)
        pygame.draw.line(self.screen, self.colors["border"], self.left_panel_rect.topright, self.left_panel_rect.bottomright, 1)
        pygame.draw.line(self.screen, self.colors["border"], self.right_panel_rect.topleft, self.right_panel_rect.bottomleft, 1)

    def _draw_left_panel(self):
        self.screen.blit(
            self.font_title.render("Scenario Designer", True, self.colors["text"]),
            self.left_header_title_pos,
        )
        self.screen.blit(
            self.font_small.render("Configure a scenario, then place it on the map.", True, self.colors["muted"]),
            self.left_header_subtitle_pos,
        )
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(self.left_body_rect)

        self._draw_soft_section(self.section_rects["scene_name"])
        self._draw_section_header(
            "Scene Name",
            self.section_rects["scene_name"].x + self.section_pad,
            self.section_rects["scene_name"].y + self.section_pad,
        )
        self.scene_name.draw(self.screen)

        self._draw_soft_section(self.section_rects["setup"])
        self._draw_section_header(
            "Scenario Setup",
            self.section_rects["setup"].x + self.section_pad,
            self.section_rects["setup"].y + self.section_pad,
        )
        scenario_label_y = self.scenario_selector.rect.y - self.spacing_xs - self.font_small.get_height()
        level_label_y = self.level_selector.rect.y - self.spacing_xs - self.font_small.get_height()
        self._draw_label("Scenario", self.scenario_selector.rect.x, scenario_label_y, small=True)
        self.scenario_selector.draw(self.screen)
        self._draw_label("Difficulty", self.level_selector.rect.x, level_label_y, small=True)
        self.level_selector.draw(self.screen)

        self._draw_soft_section(self.section_rects["editor"])
        self._draw_tab_button(self.editor_tab_rects["parameters"], "Parameters", active=self.editor_tab == "parameters")
        self._draw_tab_button(self.editor_tab_rects["actors"], "Actors", active=self.editor_tab == "actors")

        if self.editor_tab == "parameters":
            for field in self.active_fields:
                box = self.field_boxes[field.key]
                self._draw_label(
                    field.label,
                    box.rect.x,
                    box.rect.y - self.spacing_xs - self.font_small.get_height(),
                    small=True,
                )
                box.draw(self.screen)
        else:
            self.place_ego_btn.draw(self.screen)
            self.place_vehicle_btn.draw(self.screen)
            self.place_ped_btn.draw(self.screen)
            self.del_btn.draw(self.screen)
            self._draw_actor_rows()
            if hasattr(self, "draw_actor_detail_panel"):
                self.draw_actor_detail_panel(self.screen, self.actor_detail_rect)

        self._draw_soft_section(self.section_rects["status"])
        status_x = self.section_rects["status"].x + self.section_pad
        status_y = self.section_rects["status"].y + self.section_pad
        self._draw_section_header("Status", status_x, status_y)
        summary_y = status_y + self.font_section.get_height() + self.spacing_sm
        for idx, line in enumerate(self.status_summary_lines()):
            self.screen.blit(
                self.font.render(line, True, self.colors["text"]),
                (
                    status_x,
                    summary_y + idx * (self.font.get_height() + self.spacing_sm),
                ),
            )

        self.screen.set_clip(previous_clip)

    def _draw_actor_rows(self):
        actor_colors = {
            "agent": (29, 110, 235),
            "vehicle": (0, 118, 182),
            "pedestrian": (201, 63, 63),
        }
        for idx, row_rect in enumerate(self.actor_row_rects):
            is_selected = idx == self.selected_actor_index
            bg = (227, 238, 255) if is_selected else (246, 247, 249)
            border = self.colors["accent"] if is_selected else (204, 209, 216)
            pygame.draw.rect(self.screen, bg, row_rect, border_radius=self.button_radius)
            pygame.draw.rect(self.screen, border, row_rect, 1, border_radius=self.button_radius)
            if idx >= len(self.designed_actors):
                continue
            actor = self.designed_actors[idx]
            label = actor["type"].replace("agent", "ego").title()
            role = actor.get("role")
            start = actor["start"]
            end = actor["end"]
            prefix = f"{label}/{role}" if role and role not in {actor["type"], "ego"} else label
            text = f"{prefix}: ({start[0]}, {start[1]}) -> ({end[0]}, {end[1]})"
            dot_color = actor_colors.get(actor["type"], self.colors["accent"])
            pygame.draw.circle(self.screen, dot_color, (row_rect.x + 10, row_rect.centery), 4)
            surf = self.font_small.render(text, True, self.colors["text"])
            self.screen.blit(surf, (row_rect.x + 20, row_rect.y + (row_rect.height - surf.get_height()) // 2))

    def _draw_actor_overlay(self):
        actor_colors = {
            "agent": (29, 110, 235),
            "vehicle": (0, 118, 182),
            "pedestrian": (201, 63, 63),
        }
        for idx, actor in enumerate(self.designed_actors):
            start = self.map_to_screen_pos(actor["start"])
            end = self.map_to_screen_pos(actor["end"])
            if start is None or end is None:
                continue
            color = actor_colors.get(actor["type"], self.colors["accent"])
            width = 4 if idx == self.selected_actor_index else 2
            pygame.draw.line(self.screen, color, start, end, width)
            pygame.draw.circle(self.screen, (50, 180, 80), start, 6 if idx == self.selected_actor_index else 5)
            pygame.draw.circle(self.screen, (215, 80, 80), end, 6 if idx == self.selected_actor_index else 5)
        if self.pending_actor_start is not None:
            start = self.map_to_screen_pos(self.screen_to_map_pos(self.pending_actor_start))
            if start is not None:
                pygame.draw.circle(self.screen, self.colors["accent"], start, 6)

    def _draw_center_canvas(self):
        frame_rect = self.map_rect.inflate(self.frame_pad, self.frame_pad)
        pygame.draw.rect(self.screen, self.colors["canvas_frame"], frame_rect, border_radius=self.card_radius + 4)
        pygame.draw.rect(
            self.screen,
            self.colors["canvas_bg"],
            frame_rect.inflate(-2, -2),
            border_radius=self.card_radius + 2,
        )

        cropped_map = self.env.map.map_surface.subsurface(self._map_crop_rect)
        scaled_map = pygame.transform.scale(cropped_map, self.map_rect.size)
        self.screen.blit(scaled_map, self.map_rect.topleft)
        self._draw_actor_overlay()

        help_text = self.font_small.render(self.current_map_help_text(), True, self.colors["muted"])
        self.screen.blit(help_text, (self.map_rect.x, self.timeline_rect.y - self.font_small.get_height() - self.spacing_sm))
        self.draw_timeline()

    def _draw_right_panel(self):
        self.screen.blit(
            self.font_title.render("Preview", True, self.colors["text"]),
            self.right_header_title_pos,
        )
        self.screen.blit(
            self.font_small.render("Run, inspect, and save the current setup.", True, self.colors["muted"]),
            self.right_header_subtitle_pos,
        )

        self.anchor_btn.draw(self.screen)
        self.play_btn.draw(self.screen)
        self.add_rdm_actor_btn.draw(self.screen)
        self._draw_card_rect(self.saved_scenes_rect)
        list_title_x = self.saved_scenes_rect.x + self.section_pad
        list_title_y = self.saved_scenes_rect.y + self.section_pad
        self.screen.blit(
            self.font_section.render("Saved Scenes", True, self.colors["text"]),
            (list_title_x, list_title_y),
        )
        self.listbox.draw(self.screen)
        self.save_btn.draw(self.screen)
        self.load_btn.draw(self.screen)

    def update_frame_from_mouse(self, mouse_x):
        rel_x = max(0, min(mouse_x - self.timeline_rect.x, self.timeline_rect.width))
        self.current_frame = int((rel_x / max(1, self.timeline_rect.width)) * self.total_frames)

    def _draw_card(self, x, y, width, height):
        rect = pygame.Rect(x, y, width, height)
        self._draw_card_rect(rect)

    def _draw_card_rect(self, rect):
        pygame.draw.rect(self.screen, self.colors["card_bg"], rect, border_radius=self.card_radius)
        pygame.draw.rect(self.screen, self.colors["border"], rect, width=1, border_radius=self.card_radius)

    def _draw_soft_section(self, rect):
        pygame.draw.rect(self.screen, (250, 251, 252), rect, border_radius=self.section_radius)
        pygame.draw.rect(self.screen, (214, 219, 226), rect, width=1, border_radius=self.section_radius)

    def _draw_label(self, text, x, y, small=False):
        font = self.font_small if small else self.font
        self.screen.blit(font.render(text, True, self.colors["muted"]), (x, y))

    def _draw_section_header(self, text, x, y):
        self.screen.blit(self.font_section.render(text, True, self.colors["text"]), (x, y))

    def _draw_tab_button(self, rect, text, active=False):
        bg = self.colors["accent_soft"] if active else (246, 247, 249)
        border = self.colors["accent"] if active else (204, 209, 216)
        text_color = (18, 73, 170) if active else self.colors["text"]
        pygame.draw.rect(self.screen, bg, rect, border_radius=self.button_radius)
        pygame.draw.rect(self.screen, border, rect, 1 if not active else 2, border_radius=self.button_radius)
        label = self.font.render(text, True, text_color)
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_disabled_button(self, rect, text):
        pygame.draw.rect(self.screen, self.colors["success_soft"], rect, border_radius=self.button_radius)
        pygame.draw.rect(self.screen, self.colors["border"], rect, width=1, border_radius=self.button_radius)
        label = self.font.render(text, True, self.colors["success"])
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_scroll_indicator(self):
        track_w = max(4, int(round(6 * self.ui_scale)))
        track_rect = pygame.Rect(
            self.left_panel_rect.right - self.spacing_sm,
            self.left_body_rect.y + self.spacing_sm,
            track_w,
            self.left_body_rect.height - 2 * self.spacing_sm,
        )
        pygame.draw.rect(self.screen, (220, 225, 232), track_rect, border_radius=track_w // 2)
        max_scroll = max(1, self.left_content_height - self.left_body_rect.height)
        thumb_h = max(
            int(track_rect.height * (self.left_body_rect.height / max(1, self.left_content_height))),
            self.layout_cfg.scroll_thumb_min,
        )
        thumb_y = track_rect.y + int((track_rect.height - thumb_h) * (self.left_scroll_offset / max_scroll))
        thumb_rect = pygame.Rect(track_rect.x, thumb_y, track_rect.width, thumb_h)
        pygame.draw.rect(self.screen, (178, 186, 197), thumb_rect, border_radius=track_w // 2)

    def _draw_wrapped_text(self, text, rect, font, color):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            probe = word if not current else f"{current} {word}"
            if font.size(probe)[0] <= rect.width:
                current = probe
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        for idx, line in enumerate(lines[:3]):
            surf = font.render(line, True, color)
            self.screen.blit(surf, (rect.x, rect.y + idx * (font.get_height() + 2)))

    def _compute_map_crop_rect(self, surface):
        arr = pygame.surfarray.array3d(surface)
        bg = arr[0, 0].astype(np.int16)
        diff = np.abs(arr.astype(np.int16) - bg).sum(axis=2)
        mask = diff > self.layout_cfg.crop_diff_threshold

        # Ignore the low-information top strip by looking for the first sustained
        # band of wide horizontal content rather than isolated corner pixels.
        row_activity_ratio = mask.sum(axis=0) / max(1, surface.get_width())
        sustained_threshold = self.layout_cfg.crop_top_activity_threshold
        sustained_rows = np.where(row_activity_ratio > sustained_threshold)[0]
        if sustained_rows.size > 0:
            top_cut = int(sustained_rows[0])
            max_trim = int(surface.get_height() * self.layout_cfg.crop_top_trim_ratio_max)
            top_cut = min(top_cut, max_trim)
            if top_cut > 0:
                mask[:, :top_cut] = False

        coords = np.argwhere(mask)
        if coords.size == 0:
            return pygame.Rect(0, 0, surface.get_width(), surface.get_height())

        xs = coords[:, 0]
        ys = coords[:, 1]
        pad = self.layout_cfg.crop_padding
        xmin = max(0, int(xs.min()) - pad)
        xmax = min(surface.get_width() - 1, int(xs.max()) + pad)
        ymin = max(0, int(ys.min()) - pad)
        ymax = min(surface.get_height() - 1, int(ys.max()) + pad)
        return pygame.Rect(xmin, ymin, max(1, xmax - xmin + 1), max(1, ymax - ymin + 1))

    @property
    def scenario_spec(self):
        return get_scenario_spec(self.scenario_selector.selection)


def np_to_surface(arr):
    return pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
