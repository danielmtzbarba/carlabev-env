import pygame

from bicycle import Robot
import utils

from RRT_Star import RRTMap


running = True


def navigate(map, path):
    lasttime = pygame.time.get_ticks()
    pygame.init()

    def robot_simulate(dt, event=None):
        robot.move(dt, event=event)
        robot.draw(map.map)

    path = path
    # path.reverse()

    x_path = []
    y_path = []
    for loc in path:
        loc = utils.scale_coords(loc, map.scale)
        x_path.append(loc[0])
        y_path.append(loc[1])

    map.drawPath(path, (255, 0, 0), 5)

    xypath = (x_path, y_path)
    print(xypath)
    robot = Robot(map.start_position, xypath)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if not running:
                pygame.quit()
        pygame.display.update()
        dt = (pygame.time.get_ticks() - lasttime) / 1000
        lasttime = pygame.time.get_ticks()
        map.drawMap()
        robot_simulate(dt)
        running = robot.carfoundgoal()


size = 128

if __name__ == "__main__":
    path = utils.target_locations
    goal = utils.scale_coords(path[len(path) - 1], int(1024 / size))[:-1]
    start = utils.get_spawn_locations(size)[:-1]
    print(f"start: {start} - goal: {goal}")
    map = RRTMap(start, goal, size)
    navigate(map, path)
