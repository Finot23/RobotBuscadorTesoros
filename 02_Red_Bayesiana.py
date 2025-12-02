
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Definir la estructura (Causas -> Efectos)
modelo_robot = DiscreteBayesianNetwork([
    ('Tesoro', 'Vibracion'),
    ('SueloMojado', 'Vibracion'),
    ('Tesoro', 'Profundidad'),
    ('Vibracion', 'Cavar'),
    ('Profundidad', 'Cavar')
])

# P(Tesoro)
cpd_tesoro = TabularCPD(
    variable='Tesoro', variable_card=2,
    values=[[0.01], [0.99]], # P(Si)=1%, P(No)=99% (Es raro encontrar un tesoro)
    state_names={'Tesoro': ['Si', 'No']}
)

# P(SueloMojado)
cpd_suelo = TabularCPD(
    variable='SueloMojado', variable_card=2,
    values=[[0.3], [0.7]], # P(Si)=30%, P(No)=70%
    state_names={'SueloMojado': ['Si', 'No']}
)

# P(Profundidad | Tesoro)
# Profundidad_card=2: [Poca, Mucha]
# Tesoro_card=2: [Si, No]
cpd_profundidad = TabularCPD(
    variable='Profundidad', variable_card=2,
    values=[[0.3, 0.8],  # P(Poca | Tesoro=Si) y P(Poca | Tesoro=No)
            [0.7, 0.2]],  # P(Mucha | Tesoro=Si) y P(Mucha | Tesoro=No)
    evidence=['Tesoro'], evidence_card=[2],
    state_names={'Profundidad': ['Poca', 'Mucha'], 'Tesoro': ['Si', 'No']}
)
# P(Vibracion | Tesoro, SueloMojado)
# Vibracion_card=3: [Alta, Media, Baja]
# Tesoro_card=2: [Si, No]
# SueloMojado_card=2: [Si, No]
cpd_vibracion = TabularCPD(
    variable='Vibracion', variable_card=3,
    values=[
        # T=Si, SM=Si | T=No, SM=Si | T=Si, SM=No | T=No, SM=No
        [0.8,   0.2,     0.9,     0.05],  # P(Alta | ...)
        [0.1,   0.5,     0.05,    0.20],  # P(Media | ...)
        [0.1,   0.3,     0.05,    0.75]   # P(Baja | ...)
    ],
    evidence=['Tesoro', 'SueloMojado'], evidence_card=[2, 2],
    state_names={'Vibracion': ['Alta', 'Media', 'Baja'],
                 'Tesoro': ['Si', 'No'],
                 'SueloMojado': ['Si', 'No']}
)

# P(Cavar | Vibracion, Profundidad)
# Cavar_card=2: [Si, No]
# Vibracion_card=3: [Alta, Media, Baja]
# Profundidad_card=2: [Poca, Mucha]
cpd_cavar = TabularCPD(
    variable='Cavar', variable_card=2,
    values=[
        # V=Alta, P=Poca | V=Media, P=Poca | V=Baja, P=Poca | V=Alta, P=Mucha | V=Media, P=Mucha | V=Baja, P=Mucha
        [0.95,   0.70,      0.10,     0.60,      0.30,        0.01],  # P(Cavar=Si | ...)
        [0.05,   0.30,      0.90,     0.40,      0.70,        0.99]   # P(Cavar=No | ...)
    ],
    evidence=['Vibracion', 'Profundidad'], evidence_card=[3, 2],
    state_names={'Cavar': ['Si', 'No'],
                 'Vibracion': ['Alta', 'Media', 'Baja'],
                 'Profundidad': ['Poca', 'Mucha']}
)
# 4. Agregar CPDs al modelo
modelo_robot.add_cpds(cpd_tesoro, cpd_suelo, cpd_profundidad, cpd_vibracion, cpd_cavar)

# 5. Crear objeto de inferencia
inferencia = VariableElimination(modelo_robot)

## --- Escenario 1: Decisión de Cavar (Consulta Predictiva) ---
# ¿Cuál es la probabilidad de que el robot decida cavar? P(Cavar)
prob_cavar_apriori = inferencia.query(variables=['Cavar'])
print("### 1. Probabilidad a priori de Cavar ###")
print(prob_cavar_apriori)


## --- Escenario 2: Diagnóstico (Consulta a Posteriori/Condicional) ---
# El robot recibe una VIBRACIÓN ALTA y el SUELO NO ESTÁ MOJADO.
# ¿Cuál es la probabilidad real de que haya un Tesoro? P(Tesoro | Vibracion=Alta, SueloMojado=No)
prob_tesoro_dado_evidencia = inferencia.query(
    variables=['Tesoro'],
    evidence={'Vibracion': 'Alta', 'SueloMojado': 'No'}
)
print("\n### 2. Probabilidad de Tesoro dada la Evidencia ###")
print(prob_tesoro_dado_evidencia)

## --- Escenario 3: Decisión con Riesgo (Consulta de Utilidad) ---
# Si la Profundidad es Mucha y la Vibración es Media, ¿debe Cavar? P(Cavar | Profundidad=Mucha, Vibracion=Media)
prob_cavar_dado_riesgo = inferencia.query(
    variables=['Cavar'],
    evidence={'Profundidad': 'Mucha', 'Vibracion': 'Media'}
)
print("\n### 3. Probabilidad de Cavar dado un Escenario de Riesgo ###")
print(prob_cavar_dado_riesgo)
