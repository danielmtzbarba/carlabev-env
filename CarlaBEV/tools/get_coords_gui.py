from CarlaBEV.envs.utils import load_planning_map, load_map
import matplotlib.pyplot as plt

map = load_planning_map()
rgbmap, _ = load_map(size=1024)

fig, ax = plt.subplots()
ax.imshow(rgbmap)
plt.title("Click on the image to get coordinates. Close the window when done.")

points = []
idx = 0

while True:
    coords = plt.ginput(n=1, timeout=3)
    if not coords:
        continue
    x, y = coords[0]
    points.append([(int(y), int(x))])
    ax.scatter(x, y, c='r', s=2)
    ax.annotate(str(idx), (x, y), color='k', fontsize=12, weight='bold')
    plt.draw()
    print(f"Point {idx}: (y={int(y)}, x={int(x)})")
    idx += 1

plt.show()
print()