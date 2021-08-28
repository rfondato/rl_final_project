<h1> Trabajo Final: Reinforcement Learning </h1>
<h3> Alumno: Rodrigo Fondato </h3>

<h2> Detalle de módulos y otros archivos </h2>

* arena.py -> Contiene la clase Arena, que permite jugar partidos entre dos jugadores
que implementen BasePlayer (Random, Greedy, Torch), para luego indicar el ganador
e imprimir estadísticas de cada jugador.
* custom_features_extractor.py -> Contiene los siguientes features extractors:
  * CustomBoardExtractor: Para usar con red MLP. Es esencialmente un flatten, pero con un cambio con respecto al
proyecto base: la utilización de un segundo canal para las máscaras, en vez del cálculo de movimientos válidos
en cada foward, predict y evaluate. Este extractor ignora el canal de máscaras y solo utiliza
el board que viene en el 1er canal.
Utilizar un canal de máscara me permitió acelerar mucho el entrenamiento. Los movimientos válidos
se calculan una vez por cada step y en cada environment paralelo.
Al momento del foward, evaluate, predict se trabaja directo con tensores en GPU.
  * CNNFeaturesExtractor: Pasa el tablero por una red CNN para extraer features que luego van
al MLP defecto de PPO. La arquitectura tienen 3 capas convolucionales 2d.
* custom_policies.py -> Contiene el CustomActorCritic. Una diferencia respecto al base es la
utilización del canal 2 para calcular los masked logits, como se menciona en CustomBoardExtractor.
* dynamic_programming.py -> Presente en proyecto base. No hay modificaciones.
* learn.py -> Script que puede ser ejecutado por consola y facilita el entrenamiento. Permite pasar
argumentos con parámetros del entrenamiento, especificar un modelo a cargar para continuar un entrenamiento, especificar
oponentes y de que archivos cargarlos, etc. 
Ejecutar "python learn.py -h" para mayor detalle.
* mcts.py -> Presente en proyecto base. No tiene modificaciones pero fue reescrito en multi_process_mcts.py
  (mas detalle abajo)
* multi_env.py -> Se hicieron modificaciones para: 
  * Poder correr múltiples ambientes paralelos con diferentes argumentos: Esto permite entrenar contra varios
oponentes distintos al mismo tiempo (por ejemplo 8 procesos con 8 estrategias distintas)
  * SelfPlayEnv fue modificado para devolver los movimientos válidos en el 2do canal.
* multi_process_mcts.py => Contiene una nueva clase MultiProcessMonteCarlo que permite:
  * Correr múltiples partidos por cada nodo a explorar en paralelo (procesos distintos).
  * Usar nivel de profundidad como condición de corte. Esto permite explorar balanceadamente
todas las ramas.
* play.py -> Script que permite jugar un partido entre 2 jugadores que implementen BasePlayer.
Contiene diferentes opciones parametrizables para configurar a cada jugador, la cantidad de partidos, etc.
Ejecutar "python play.py -h" para más detalle.
* play_tournament.py -> Script que permite configurar mediante argumentos y jugar un torneo entre varios jugadores,
imprimiendo luego la tabla de posiciones y guardando en logs los detalles de cada encuentro.
Ejecutar "python play_tournament.py -h" para más detalle.
* players.py => Contiene las diferentes estrategias (jugadores): RandomPlayer, GreedyPlayer, TorchPlayer, etc.
Fue modificada para incluir: Mi implementación de TorchPlayer con el nuevo monte carlo, 
una interfaz común (BasePlayer) y agregar métodos útiles para el torneo y arena.
* print_utils.py -> Utilitarios para imprimir colores y estilos en consola y notebooks.
* reversi_model.py -> Contiene una clase de alto nivel: CustomReversiModel (nueva) que permite encapsular todo lo referido
al entrenamiento con PPO. Utilizada por el script learn.py.
* rfondato.py -> Ignorar. Archivo que contiene todas las clases custom necesarias en un solo módulo 
para levantar y correr el modelo rfondato.zip (el mejor modelo que pude entrenar).
La idea es tener un único módulo fácil de pasar (junto con el zip) para competir.
* reversi_state.py -> Presente en el proyecto base. Modificada para agregar una nueva clase: CustomReversiState
utilizada por el MultiProcessMonteCarlo.
* tournament.py -> Contiene las clases Match y Tournament, utilizadas por play_tournament.py para modelar y jugar
el torneo.
* tree_search.py -> Presente en proyecto base y no fue modificado.
* ./selected_models -> Carpeta que contiene los .zips con los distintos modelos que fuí entrenando.
Se puede utilizar de entrada en el script play_tournament.py (Ver 010_Torneo.ipynb)
* ./selected_logs -> Carpeta con logs de tensorboard para los modelos seleccionados.
* ./modelo_competencia -> Ver más abajo.

<h2> Nuevas notebooks: </h2>

Se agregaron 2 notebooks nuevas:
* 009_Analisis => Contiene el análisis de cada paso que fuí siguiendo para mejorar el entrenamiento
y conclusiones.
* 010_Torneo => Contiene la ejecución de un torneo entre todos los modelos y conclusiones.

<h3> Aclaración para competencia: </h3>

La carpeta ./modelo_competencia contiene un archivo rfondato.py y un archivo rfondato.zip.
El primero implementa todas las clases custom necesarias para cargar el modelo PPO y la clase RFondatoPlayer,
que modeliza mi jugador (sin MCTS, simplificada). El .zip es el modelo PPO grabado por stable-baselines3, que es cargado automáticamente por
RFondatoPlayer.
Seguir las instrucciones en rfondato.py.
