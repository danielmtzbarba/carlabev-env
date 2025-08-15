import pygame
import numpy as np

class Node(object):
    def __init__(self, id, position, lane=None):
        self.id, self.lane = id, lane
        self._x = int(position[0])
        self._y = int(position[1])
        self.draw_x = self._x + cfg.offx
        self.draw_y = self._y + cfg.offy
        self.btn = pygame.Rect(self.draw_x, self.draw_y,  3, 3)
        self.color = None 
    
    def reset(self):
        self.color = None 
    
    def render(self, screen, color=None):
        if color is not None:
            self.color = color

        if self.color is not None:
            pygame.draw.rect(screen, self.color, self.btn)
    
    def clicked(self, event):
        if self.btn.collidepoint(event.pos):
            self.color = cfg.red 
            return True
    
    @property
    def scaled_pos(self):
        return [self._x, self._y]

    @property
    def pos(self):
        return [self.draw_x, self.draw_y]

class Actor(object):
    def __init__(self, id, start_node, end_node):
        self.id = id
        self.start_node = start_node
        self.end_node = end_node
        self.rx, self.ry = [], [] 
        self.path = []
        self.selected = False
    
    def set_route_wp(self, node_id, x, y):
        self.rx.append(x)
        self.ry.append(y)
        pos = np.array([x, y])
        self.path.append(Node(node_id, pos))

    def draw(self, screen):
        if self.selected:
            self.start_node.render(screen, cfg.green)
            self.end_node.render(screen, cfg.red)
            for node in self.path:
                node.render(screen, cfg.blue)

    @property
    def data(self):
        return [None, self.id, self.start_node.scaled_pos, self.end_node.scaled_pos, self.rx, self.ry]
# -----------------------------------------
        
class Scene(object):
    cols = ["scene_id", "class", "start", "goal", "rx", "ry"]
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._scene_id = '' 
        self._scene_data = pd.DataFrame(data=[], columns=self.cols)
        self._actors = {
            'agent': [],
            'vehicle': [],
            'pedestrian': []
        }
        self._idx = 0

    def add_actor(self, actor_type: str, start_node, end_node):
#        id=f'{actor_type}-{len(self._actors[actor_type.lower()])}'
        actor = Actor(actor_type.lower(), start_node, end_node)
        self._actors[actor_type.lower()].append(actor)
        self._idx += 1 
        return actor
    
    def draw_scene(self):
        for actor_type, actors in self._actors.items():
            for actor in actors:
                actor.draw(self.screen)
        
    def get_scene_df(self, scene_id):
        i = 0
        df = pd.DataFrame(data=[], columns=self.cols)
        for actor_type in self._actors.keys():
            for actor in self._actors[actor_type]:
                data = actor.data
                data[0] = scene_id
                df.loc[i] = data
                rx = df.loc[i, "rx"]
                ry = df.loc[i, "ry"]
                df.loc[i, "rx"] = [8*int(x) for x in rx]
                df.loc[i, "ry"] = [8*int(y) for y in ry]
                i+=1
                                
        df.astype({'rx': "object", 'ry': "object"}).dtypes
        self._scene_data = df
        return self._scene_data


class Map(Scene):
    def __init__(self, screen, size=1024) -> None:
        Scene.__init__(self)
        _,  self._map_img, _ = load_map(size)
        self.screen = screen
        self.size = size  
        #
        self.planner_ped = GraphPlanner(os.path.join(asset_path, "Town01/town01.pkl"))
        self.planner_car = GraphPlanner(os.path.join(asset_path, "Town01/town01-vehicles.pkl"))
        #
        self.planner = {
            "vehicle": self.planner_car,
            "pedestrian": self.planner_ped
        }

    def render(self, map_sur):
        # Draw map
        if map_sur is not None:
            self.screen.blit(map_sur, (cfg.offx, cfg.offy))
        else:
            self.screen.blit(self._map_img, (cfg.offx, cfg.offy))
        # Draw scene
        self.draw_scene()
    
    def get_random_node(self, actor_type):
        actor_cls = "sidewalk" if actor_type == "Pedestrian" else "vehicle"
        planner_id = "pedestrian" if actor_type == "Pedestrian" else "vehicle"
        planner = self.planner[planner_id]
        rdm_node_id = planner.get_random_node(actor_cls)
        pos = planner.get_node_pos(rdm_node_id)
        return  Node(rdm_node_id, pos)
    
    def select_node(self, event, lane, actor):
        min_dist = float('inf')
        closest_node = None
        click_pos = np.array([event.pos[0], event.pos[1]]) 
        click_pos += np.array([-cfg.offx, -cfg.offy])
        
        planner_id = "pedestrian" if actor.lower() == "pedestrian" else "vehicle"
        planner = self.planner[planner_id]
        node = planner.get_closest_node(click_pos * 8, None) 

        node_pos = np.array(planner.G.nodes[node]['pos'])
        dist = np.linalg.norm(8 * click_pos - node_pos)

        if dist < min_dist:
            min_dist = dist
            pos = planner.get_node_pos(node)/8
            closest_node = Node(node, pos, lane=None) 

        return closest_node
    
    def find_route(self, actor, lane):
        planner_id = "pedestrian" if actor.id == "pedestrian" else "vehicle"
        planner = self.planner[planner_id]
        start, end = actor.start_node, actor.end_node
        #
        if start.lane == end.lane:
            path, _= planner.find_path(start.id, end.id, actor.id)

            rx, ry, path_pos = [], [], []
            for node_id in path[1:-1]:
                pos_scaled = planner.G.nodes[node_id]['pos']
                x, y = pos_scaled[0]/8, pos_scaled[1]/8
                actor.set_route_wp(node_id, x, y)
        return actor
