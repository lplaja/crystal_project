import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 1. Definimos el estilo por defecto
try:
    plt.style.use('obsidian')
except:
    pass # Si no encuentra el .mplstyle no rompe el inicio

# 2. Registramos el Colormap globalmente
colors = ["#FFFFFF", "#D8FCBA", "#8FA97A", "#71964D", "#4A6741"]
obsidian_cmap = LinearSegmentedColormap.from_list("obsidian", colors)

# Esto lo registra para que cualquier gráfico pueda usar cmap='obsidian'
plt.colormaps.register(name='obsidian', cmap=obsidian_cmap)
plt.colormaps.register(name='obsidian_r', cmap=obsidian_cmap.reversed())