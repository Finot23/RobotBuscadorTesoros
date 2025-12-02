import heapq
import math
import pygame
import random
import time

# colores
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)
cyan = (0, 255, 255)
black = (0, 0, 0)
white = (255, 255, 255)
purple = (128, 0, 128)

agent_colors = [red, green, blue, yellow] 

rect_size = 72          
display_size = 720      

pygame.init()
display = pygame.display.set_mode((display_size, display_size))
pygame.display.set_caption("MDP - tablero 10x10")

framerate = 60
clock = pygame.time.Clock()

font = pygame.font.Font('freesansbold.ttf', 10)

try:
    background_image = pygame.image.load('tablero10.jpeg')
    background_image = pygame.transform.scale(background_image, (display_size, display_size)) 
except pygame.error as e:
    background_image = pygame.Surface((display_size, display_size))
    background_image.fill(white)

# funciones auxiliares

def dis2(a, b):
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def rotate_pt(p, c, theta): 
    sin = math.sin(theta)
    cos = math.cos(theta)
    return cos*(p[0]-c[0]) - sin*(p[1]-c[1]) + c[0], sin*(p[0]-c[0]) + cos*(p[1]-c[1]) + c[1]

def scale_segment(p0, p1, u):
    (x0, y0), (x1, y1) = p0, p1  
    dx, dy = (x1 - x0), (y1 - y0)  
    d = math.sqrt(dx * dx + dy * dy)  
    if d == 0: return p0, p0
    return p0, (x0 + dx*(u/d), y0 + dy*(u/d))

def draw_arrow(screen, color, p1, p2, width):
    pygame.draw.line(screen, color, p1, p2, width)
    
    a, b = scale_segment(p1, p2, dis2(p1, p2) - width * 2) 
    
    left = rotate_pt(b, p2, math.pi / 15)  
    right = rotate_pt(b, p2, -math.pi / 15)  
    pygame.draw.line(screen, color, left, p2, width)  
    pygame.draw.line(screen, color, right, p2, width)

# mdp configuracion

grid_width = display.get_width() // rect_size
grid_height = display.get_height() // rect_size

gamma = 0.95     
goal_reward = 100.0
move_cost = -1.0 
obstacle_penalty = -50.0 
actions = {'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1)}
action_names = list(actions.keys())

current_method = 'vi'

v = [[0.0 for _ in range(grid_width)] for _ in range(grid_height)]
pi = [['' for _ in range(grid_width)] for _ in range(grid_height)]

def cell_to_coords(y, x):
    return x * rect_size, y * rect_size

def coords_to_cell(px, py):
    return py // rect_size, px // rect_size

# meta y obstaculos
goal_state = (4, 4) 

obstacles = set([
    (0, 0), (0, 2), (0, 5), (0, 7), (1, 4), (1, 9), (2, 2), (2, 7), 
    (4, 2), (4, 7), (5, 0), (5, 2), (5, 5), (5, 7), (6, 4), (6, 9), 
    (7, 2), (7, 7), (9, 2), (9, 7)
])

def get_reward(y, x):
    if (y, x) == goal_state:
        return goal_reward
    if (y, x) in obstacles:
        return obstacle_penalty
    return move_cost 

def clip_to_grid(ny, nx, cy, cx):
    if 0 <= ny < grid_height and 0 <= nx < grid_width:
        if (ny, nx) in obstacles:
            return cy, cx 
        return ny, nx
    return cy, cx 

def get_transitions(y, x, action):
    if (y, x) == goal_state or (y, x) in obstacles:
        return [(1.0, (y, x))]

    main_dy, main_dx = actions[action]
    lateral_actions = []
    if action in ['u', 'd']:
        lateral_actions = [actions['l'], actions['r']]
    elif action in ['l', 'r']:
        lateral_actions = [actions['u'], actions['d']]
    
    transitions = []

    ny1, nx1 = y + main_dy, x + main_dx
    transitions.append((0.8, clip_to_grid(ny1, nx1, y, x)))

    for lat_dy, lat_dx in lateral_actions:
        ny2, nx2 = y + lat_dy, x + lat_dx
        transitions.append((0.1, clip_to_grid(ny2, nx2, y, x)))

    return transitions

# algoritmos de solucion

def value_iteration(epsilon=0.01):
    global v, pi
    
    while True:
        delta = 0
        v_new = [[0.0 for _ in range(grid_width)] for _ in range(grid_height)]
        
        for y in range(grid_height):
            for x in range(grid_width):
                
                if (y, x) == goal_state:
                    v_new[y][x] = goal_reward
                    continue
                
                if (y, x) in obstacles:
                    v_new[y][x] = obstacle_penalty
                    continue

                r = get_reward(y, x)
                max_q = -float('inf')
                best_action = ''
                
                for action in action_names:
                    expected_value = 0
                    
                    for prob, (ny, nx) in get_transitions(y, x, action):
                        expected_value += prob * v[ny][nx]
                    
                    q_s_a = r + gamma * expected_value
                    
                    if q_s_a > max_q:
                        max_q = q_s_a
                        best_action = action

                v_new[y][x] = max_q
                pi[y][x] = best_action
                
                delta = max(delta, abs(v_new[y][x] - v[y][x]))

        v = v_new
        if delta < epsilon * (1 - gamma) / gamma:
            return

def policy_evaluation(pi_current, v_current, theta=0.001):
    while True:
        delta = 0
        v_new = [[v_val for v_val in row] for row in v_current]
        
        for y in range(grid_height):
            for x in range(grid_width):
                
                if (y, x) == goal_state or (y, x) in obstacles:
                    continue 

                action = pi_current[y][x]
                if not action: continue

                r = get_reward(y, x)
                expected_value = 0
                
                for prob, (ny, nx) in get_transitions(y, x, action):
                    expected_value += prob * v_current[ny][nx]
                
                v_s = r + gamma * expected_value
                
                delta = max(delta, abs(v_s - v_current[y][x]))
                v_new[y][x] = v_s
                
        v_current = v_new
        if delta < theta:
            return v_current

def policy_improvement(v_evaluated, pi_current):
    policy_stable = True 
    pi_new = [[p for p in row] for row in pi_current]
    
    for y in range(grid_height):
        for x in range(grid_width):
            
            if (y, x) == goal_state or (y, x) in obstacles:
                continue
                
            old_action = pi_current[y][x]
            max_q = -float('inf')
            best_action = old_action
            r = get_reward(y, x)
            
            for action in action_names:
                expected_value = 0
                for prob, (ny, nx) in get_transitions(y, x, action):
                    expected_value += prob * v_evaluated[ny][nx]
                
                q_s_a = r + gamma * expected_value
                
                if q_s_a > max_q:
                    max_q = q_s_a
                    best_action = action
            
            pi_new[y][x] = best_action
            
            if old_action != best_action:
                policy_stable = False 
                
    return pi_new, policy_stable

def policy_iteration():
    global v, pi
    
    v = [[0.0 for _ in range(grid_width)] for _ in range(grid_height)] 
    pi = [[random.choice(action_names) if (y,x) != goal_state and (y,x) not in obstacles else '' for x in range(grid_width)] for y in range(grid_height)]

    while True:
        v = policy_evaluation(pi, v)
        pi_new, policy_stable = policy_improvement(v, pi)
        pi = pi_new
        
        if policy_stable:
            return

# conf de los multiples agentes

agent_radius = rect_size / 2 * 0.8
agent_speed = 1 

class Agent: 
    def __init__(self, start_cell, destination_cell, color, priority, name):
        self.cy, self.cx = start_cell 
        self.dest_y, self.dest_x = destination_cell
        self.color = color
        self.priority = priority 
        self.name = name
        self.px, self.py = cell_to_coords(self.cy, self.cx) 
        self.target_px, self.target_py = self.px, self.py
        self.is_moving = False 
        self.finished = False 

agents_setup = [
    ((0, 1), goal_state, 4, 'a'), 
    ((2, 8), goal_state, 3, 'b'),  
    ((8, 8), goal_state, 2, 'c'),
    ((9, 3), goal_state, 1, 'd') 
]

agents = [Agent(setup[0], setup[1], agent_colors[i % len(agent_colors)], setup[2], setup[3]) for i, setup in enumerate(agents_setup)]
agents.sort(key=lambda a: a.priority, reverse=True) 

solving_time = 0
solve_needed = True

# bucle principal
run = True
while run:
    display.blit(background_image, (0, 0)) # dibuja la imagen de fondo primero

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                current_method = 'pi' if current_method == 'vi' else 'vi'
                solve_needed = True
            if event.key == pygame.K_r:
                agents = [Agent(setup[0], setup[1], agent_colors[i % len(agent_colors)], setup[2], setup[3]) for i, setup in enumerate(agents_setup)]
                agents.sort(key=lambda a: a.priority, reverse=True)
                solve_needed = True 

    # 1. resolver el mdp
    if solve_needed:
        start_time = time.time()
        if current_method == 'vi':
            value_iteration()
        elif current_method == 'pi':
            policy_iteration()
        solving_time = time.time() - start_time
        solve_needed = False

    # 2. dibujar la cuadricula y politica
    
    policy_arrows = [] 

    for y in range(grid_height):
        for x in range(grid_width):
            px, py = cell_to_coords(y, x)
            
            # Dibujar celda
            if (y, x) == goal_state:
                cell_color = yellow 
                pygame.draw.rect(display, cell_color, (px, py, rect_size, rect_size))
            
            pygame.draw.rect(display, red, (px, py, rect_size, rect_size), 1)
            
            # Dibujar la política (flecha), solo si no es un obstáculo
            policy_action = pi[y][x]
            if policy_action and (y, x) not in obstacles:
                cy, cx = y, x
                dy, dx = actions[policy_action]
                
                p1x, p1y = px + rect_size / 2, py + rect_size / 2
                
                p2x = p1x + dx * rect_size * 0.4
                p2y = p1y + dy * rect_size * 0.4
                
                draw_arrow(display, purple, (p1x, p1y), (p2x, p2y), 2)


    # 3. movimiento de multiples agentes con evasion
    occupied_cells = {} 

    for agent_obj in agents: 
        
        # animacion de movimiento
        if agent_obj.is_moving:
            current_distance = dis2((agent_obj.px, agent_obj.py), (agent_obj.target_px, agent_obj.target_py))
            
            if current_distance > agent_speed:
                ratio = agent_speed / current_distance
                agent_obj.px += (agent_obj.target_px - agent_obj.px) * ratio
                agent_obj.py += (agent_obj.target_py - agent_obj.py) * ratio
            else:
                agent_obj.px, agent_obj.py = agent_obj.target_px, agent_obj.target_py
                agent_obj.cy, agent_obj.cx = coords_to_cell(agent_obj.px, agent_obj.py)
                agent_obj.is_moving = False
                
                if (agent_obj.cy, agent_obj.cx) == (agent_obj.dest_y, agent_obj.dest_x):
                    agent_obj.finished = True
        
        # decision de movimiento
        if not agent_obj.is_moving and not agent_obj.finished:
            
            try:
                action = pi[agent_obj.cy][agent_obj.cx]
            except IndexError: 
                action = ''

            if action and action in actions:
                dy, dx = actions[action]
                
                next_cy, next_cx = agent_obj.cy + dy, agent_obj.cx + dx
                
                next_cy_real, next_cx_real = clip_to_grid(next_cy, next_cx, agent_obj.cy, agent_obj.cx)
                
                next_cell = (next_cy_real, next_cx_real)
                is_reserved = next_cell in occupied_cells and occupied_cells[next_cell] != agent_obj
                
                if not is_reserved and (next_cy_real, next_cx_real) != (agent_obj.cy, agent_obj.cx):
                    agent_obj.target_px, agent_obj.target_py = cell_to_coords(next_cy_real, next_cx_real)
                    agent_obj.is_moving = True
                    occupied_cells[next_cell] = agent_obj 
                
        # dibujar el agente
        center_x = int(agent_obj.px + rect_size / 2)
        center_y = int(agent_obj.py + rect_size / 2)
        
        if agent_obj.finished:
            pygame.draw.circle(display, agent_obj.color, (center_x, center_y), agent_radius, 1) 
        else:
            pygame.draw.circle(display, agent_obj.color, (center_x, center_y), agent_radius)
            text_pri = font.render(agent_obj.name, True, white) 
            textrect_pri = text_pri.get_rect(center=(center_x, center_y))
            display.blit(text_pri, textrect_pri)

    # Mostrar información del método
    info_text = f"metodo: {current_method} (espacio para cambiar) | tiempo solucion: {solving_time:.4f}s | r para reiniciar agentes"
    text_info = font.render(info_text, True, black, white)
    display.blit(text_info, (10, 10))
    
    pygame.display.update()
    clock.tick(framerate)

pygame.quit()