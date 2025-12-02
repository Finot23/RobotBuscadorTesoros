import heapq
import math
import pygame

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)
cyan = (0, 255, 255)
black = (0, 0, 0)
white = (255, 255, 255)

pygame.init()
display = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("Busqueda Informada")

framerate = 30
clock = pygame.time.Clock()

try:
    fo = open('pathout.txt', 'w')
except Exception as e:
    print(f"Error al abrir pathout.txt: {e}")
    fo = None 

# Variables de estado
pi = 0
lp = 0
finished = False 
pxx = [0.0, 0.0] 

font = pygame.font.Font('freesansbold.ttf', 10)
rect_size = 20

# Funciones auxiliares
def sqr(x):
    return x * x

def dis2(a, b):
    #Retorna la Distancia Euclidiana entre dos puntos (a y b)
    #usada como la funcion heuristica h(n) en A*
    return math.sqrt(sqr(b[0] - a[0]) + sqr(b[1] - a[1]))

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

def draw_arrow(screen, color, p1, p2, width, cost):
    """Dibuja una linea con cabeza de flecha y muestra el costo de la arista."""
    p1_center = (p1[0] + rect_size / 2, p1[1] + rect_size / 2)
    p2_center = (p2[0] + rect_size / 2, p2[1] + rect_size / 2)
    pygame.draw.line(screen, color, p1_center, p2_center, width)
    
    # Dibuja la cabeza de flecha
    a, b = scale_segment(p1_center, p2_center, rect_size / 2)
    left = rotate_pt(b, a, math.pi / 15)  
    right = rotate_pt(b, a, -math.pi / 15)  
    pygame.draw.line(screen, color, left, p2_center, width)  
    pygame.draw.line(screen, color, right, p2_center, width) 

    if cost != 0:
        text = font.render(str(cost), True, color, white)
        textRect = text.get_rect()
        textRect.center = ((p1_center[0] + p2_center[0]) / 2 + 15, (p1_center[1] + p2_center[1]) / 2 + 10)
        screen.blit(text, textRect)
    
# algoritmo de busqueda
def flatten(L):
    """Funcion auxiliar para aplanar la lista enlazada del camino."""
    while len(L) > 0:
        yield L[0]
        L = L[1]

def astar(G, start, end):
   
    #Implementacion del algoritmo A*  donde se usa la funcion de evaluacion f(n) = g(n) + h(n)
    #g(n) es el costo real acumulado y h(n) es la heuristica Euclidiana.
    # Cola de prioridad: (f_cost, g_cost, v1, path)
    q = [(dis2(c[start], c[end]), 0, start, ())]  
    g_costs = {start: 0} # Almacena el costo real g(n)
    visited = set()       
    
    while q:
        (f_cost, g_cost, v1, path) = heapq.heappop(q)
        
        if v1 in g_costs and g_cost > g_costs[v1]:
            continue

        if v1 not in visited:
            visited.add(v1)
            
            if v1 == end:
                return list(flatten(path))[::-1] + [v1]
            
            path = (v1, path)
            
            for (v2, cost2) in G[v1].items():
                if v2 not in visited:
                    new_g_cost = g_cost + cost2 
                    new_f_cost = new_g_cost + dis2(c[v2], c[end]) # f(v2) = g(v2) + h(v2)
                    
                    if v2 not in g_costs or new_g_cost < g_costs[v2]:
                        g_costs[v2] = new_g_cost
                        heapq.heappush(q, (new_f_cost, new_g_cost, v2, path))
                        
    return [] # Retorna lista vacia si no se encuentra camino


c = {} # Coordenadas de los nodos
g = {} # Grafo
m = {}
    
but2 = '' # marca el nodo de inicio de conexion clic central
but4 = '' # Nodo de inicio de A* o el clic Derecho
pathb = [] # Camino encontrado por A*

# para cargar la imagen
try:
    background_image = pygame.image.load('tablero.jpeg')
    background_image = pygame.transform.scale(background_image, (1280, 720))
except pygame.error as e:
    print(f"Error al cargar la imagen: {e}. Asegurese de que 'antonio.jpeg' este en el mismo directorio.")
    background_image = pygame.Surface((1280, 720))
    background_image.fill(white)
    
display.blit(background_image, (0, 0))

# bucle de pygame
run = True
while run:
    display.blit(background_image, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
        # creacion y seleccion de nodos
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            
            if event.button == 1: # Crear Nodo (Verde)
                name = str(x) + str(y)
                c[name] = (x, y)
                m[name] = 1
                g[name] = {}
                
            elif event.button == 2: # Conectar nodos (Amarillo)
                for item in c:
                    distance = dis2(c[item], (x, y))
                    if distance <= (rect_size):
                        m[item] = 2
                        but2 = item
                    else:
                        m[item] = 1
                        
            elif event.button == 3: # Iniciar Busqueda A* (Rojo)
                for item in c:
                    distance = dis2(c[item], (x, y))
                    if distance <= (rect_size):
                        m[item] = 4
                        but4 = item
                    
        # finalizacion de conexiones y busquedas                
        elif event.type == pygame.MOUSEBUTTONUP:
            x, y = event.pos
            
            if event.button == 3: # Boton derecho 
                but5 = ''
                for item in c:
                    distance = dis2(c[item], (x, y))
                    if distance <= (rect_size):
                        m[item] = 5
                        but5 = item
                        
                if but4 and but5 and but4 != but5:
                    patha = astar(g, but4, but5)
                    
                    if patha == []:
                        pathb = []
                    else:
                        pathb = patha
                        lp = len(pathb)
                        pi = 0
                        
                        po = c[pathb[0]]
                        px = c[pathb[1]]
                        pxx[0] = po[0]
                        pxx[1] = po[1]
                        finished = False
                        
                        if fo:
                            fo.write("g=" + str(g) + "\nc=" + str(c) + "\npath =" + str(patha) + "\n")
                            fo.flush()
                
            elif event.button == 2: # Boton scroll del mouse
                but3 = ''
                for item in c:
                    distance = dis2(c[item], (x, y))
                    
                    if distance <= (rect_size):
                        m[item] = 3
                        but3 = item
                        
                        # CREAR ARISTA Y CALCULAR COSTO (Distancia Euclidiana)
                        if but2 and but3 and but2 != but3:
                            b2x, b2y = c[but2]
                            b3x, b3y = c[but3]
                            distance2 = dis2((b2x, b2y), (b3x, b3y))
                            
                            costo_final = int(distance2)
                            
                            # Agregar la arista (but2 -> but3)
                            dd = {but3: costo_final}
                            g[but2].update(dd)
                            
                        # Limpiar el estado de seleccion
                        m[but2] = 1
                        if but3: m[but3] = 1
                        but2 = ''
                        
    # para dibujar los nodos y caminos
    for item in c:
        x, y = c[item]
        color = yellow
        if m[item] == 1: color = green
        elif m[item] in [2, 3]: color = yellow
        elif m[item] in [4, 5]: color = red
            
        pygame.draw.rect(display, color, (x, y, rect_size, rect_size))
        
        # Dibuja el id del nodo
        text = font.render(item, True, black, white)
        textRect = text.get_rect()
        textRect.center = (c[item][0] + 20, c[item][1] + rect_size + 5)
        display.blit(text, textRect)
    
    # DIBUJO DEL GRAFO (ARISTAS)
    for v1 in g:
        for (v2, cost2) in g[v1].items():
            p1 = c[v1]
            p2 = c[v2]
            draw_arrow(display, blue, p2, p1, 2, cost2)
            
    #DIBUJO Y ANIMACION DEL PATH (RUTA ENCONTRADA)
    if pathb:
        # Dibuja la ruta completa en verde
        p1_path = c[pathb[0]]
        for i in range(1, lp):
            p2_path = p1_path
            p1_path = c[pathb[i]]
            draw_arrow(display, green, p1_path, p2_path, 5, 0) 
            
        # Animacion del punto (agente)
        if pi < lp:
            po = c[pathb[pi-1]] if pi > 0 else c[pathb[0]]
            px = c[pathb[pi]]
            
            steps = 50 
            dx = (px[0] - po[0]) / steps
            dy = (px[1] - po[1]) / steps

            if not finished:
                pxx[0] += dx
                pxx[1] += dy
                
                tolerance = abs(dx) + abs(dy) + 0.1
                if dis2(pxx, px) < tolerance:
                    finished = True
                    pxx[0], pxx[1] = px[0], px[1]
            
            if finished:
                pi += 1
                finished = False
                
                if pi < lp:
                    po = c[pathb[pi-1]]
                    px = c[pathb[pi]]
                    pxx[0], pxx[1] = po[0], po[1]

            # Dibuja el agente (circulo cyan)
            if pi < lp:
                pygame.draw.circle(display, cyan, (int(pxx[0] + rect_size/2), int(pxx[1] + rect_size/2)), rect_size/2)
            else:
                # Resalta el nodo final y reinicia la animacion
                pygame.draw.circle(display, yellow, (int(px[0] + rect_size/2), int(px[1] + rect_size/2)), rect_size/2)
                if pi == lp:
                    pygame.time.wait(500)
                    pi = 0
                    lp = 0
                    pathb = []
    
    pygame.display.update()
    clock.tick(60)

if fo:
    fo.close()
pygame.quit()
exit()