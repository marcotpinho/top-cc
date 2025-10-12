1. Teremos um modelo que pontua arestas para cada instância dada a posição e o valor de cada nó, o número de agentes, o budget de cada agente,
 a velocidade de cada agente e a conectividade mínima entre agentes. Como labels, iremos pegar todas as arestas usadas nas soluções de pareto
 encontradas ao longo de diversas execuções e as pontuaremos de acordo com o número de usos. O modelo, então, deverá fazer uma regressão para
 adivinhar esse valor do número de usos, que corresponde à utilidade da aresta na solução final.

2. 
