import pygame
import numpy as np
import random
import time

# --- CONFIGURACIÓN GENERAL ---
GRID_SIZE = 10
CELL_SIZE = 64
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE

# Posiciones del entorno
start ={(0, 1), (3,8),(9,3),(8,8)} 
goal_state = (4, 4) 
walls = {
    (0, 0), (0, 2), (0, 5), (0, 7), (1, 4), (1, 9), (2, 2), (2, 7), 
    (4, 2), (4, 7), (5, 0), (5, 2), (5, 5), (5, 7), (6, 4), (6, 9), 
    (7, 2), (7, 7), (9, 2), (9, 7)
}

actions = ['up', 'down', 'left', 'right']
action_idx = {a: i for i, a in enumerate(actions)}

# --- PARÁMETROS Q-LEARNING ---
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 2000
Q = np.zeros((GRID_SIZE, GRID_SIZE, len(actions)))

# --- FUNCIONES AUXILIARES ---
def is_valid(state):
    r, c = state
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and state not in walls

def step(state, action):
    r, c = state
    if action == 'up':    r -= 1
    elif action == 'down':r += 1
    elif action == 'left':c -= 1
    elif action == 'right':c += 1
    next_state = (r, c)
    if not is_valid(next_state):
        return state, -5, False
    if next_state == goal_state:
        return next_state, 100, True
    return next_state, -1, False

# --- ENTRENAMIENTO Q-LEARNING ---
for ep in range(episodes):
    state = random.choice(list(start))
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = actions[np.argmax(Q[state[0], state[1]])]

        next_state, reward, done = step(state, action)
        Q[state[0], state[1], action_idx[action]] += alpha * (
            reward + gamma * np.max(Q[next_state[0], next_state[1]]) -
            Q[state[0], state[1], action_idx[action]]
        )
        state = next_state

# --- CONSTRUIR LA RUTA FINAL ---
    state = random.choice(list(start))
path = [state]
for _ in range(100):
    if state == goal_state:
        break
    action = actions[np.argmax(Q[state[0], state[1]])]
    next_state, _, _ = step(state, action)
    if next_state == state:
        break
    path.append(next_state)
    state = next_state

# --- INICIALIZAR PYGAME ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning  hacia el cofre ")

# Cargar imagen de fondo
background = pygame.image.load("mapaEQ.jpg")
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (60, 60, 60)
GOLD = (255, 200, 0)
GREEN = (0, 255, 0)
font = pygame.font.SysFont(None, 36)

# --- FUNCIÓN PARA DIBUJAR EL MAPA ---
def draw_grid(elf_pos):
    # Dibuja la imagen base del mapa
    screen.blit(background, (0, 0))
    
    # Dibuja cuadrícula y elementos
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y = c * CELL_SIZE, r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            # Muro
            if (r, c) in walls:
                pygame.draw.rect(screen, GRAY, rect, 0)
            # Meta
            elif (r, c) == goal_state:
                pygame.draw.rect(screen, GOLD, rect, 0)
                text = font.render("🎁", True, BLACK)
                screen.blit(text, (x + 16, y + 8))
            # Elfo
            elif (r, c) == elf_pos:
                pygame.draw.rect(screen, GREEN, rect, 0)
                text = font.render("🧝", True, BLACK)
                screen.blit(text, (x + 10, y + 10))
            
            # Cuadrícula (líneas finas)
            pygame.draw.rect(screen, BLACK, rect, 1)

# --- ANIMACIÓN DEL RECORRIDO ---
running = True
index = 0
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if index < len(path):
        elf_pos = path[index]
        index += 1
    else:
        elf_pos = goal_state

    draw_grid(elf_pos)
    pygame.display.flip()
    clock.tick(2)  # velocidad de animación

pygame.quit()

