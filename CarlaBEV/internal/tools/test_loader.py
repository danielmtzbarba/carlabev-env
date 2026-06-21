import os
import sys

# Add carlabev env to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from CarlaBEV.src.scenes.scenarios.jaywalk import JaywalkScenario

scenario = JaywalkScenario(map_size=128)
scene_dict, len_route = scenario.sample(level=1, config_file="CarlaBEV/assets/scenes/jaywalk_test.json")

print("Jaywalk Scenario loaded successfully!")
print(f"Agent Config: speed={scene_dict['agent'][2]}")
print(f"Pedestrians loaded: {len(scene_dict['pedestrian'])}")
p = scene_dict["pedestrian"][0]
print(f"Pedestrian speed: {p.target_speed}")
print(f"Pedestrian behavior delay: {p.behavior.start_delay}")

if isinstance(p.behavior.__class__.__name__, str):
    print(f"Loaded behavior class: {p.behavior.__class__.__name__}")
