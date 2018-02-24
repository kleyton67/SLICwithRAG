#Projeto Identificacao e Contagem de celulas usando Weka e rag_mean_color(scikit_image)
##1 - Colaboradores
Prof. Mt. Willian Paraguassu
Kleyton Sartori Leite

##2 - Problemas
###2.0 Total de superpixeis utilizados
O atual algoritmo nao consegui identificar celulas que exigijam uma verificacao de grau > 1.
####2.01 - Resoluçao
Para resolver o atual problema, criar tuplas e colocar dentro das mesmas tuplas com nodos celulas, 
onde cada celula eh representado por uma tupla, para isso, para cada nodo, analisar se é uma celula
ou parte para assim, verificar se um de seus adjacentes ja esta contabilizado.
###2.1 - Treinamentos perfeito para qualquer dimensao de imagem?
Verificou-se erros para treinamentos de imagens de resoluções diferentes.

##3 - Resolvido
###3.0 - Acesso da informacao do nodo
Normalmente as funcoes da rag, colocam apenas o peso entre os nodos, assim nas funcoes de adjacencia
sao retornados vertices, com peso de suas arestas, porem isso pode ser corrigido utilizando o metodo
node e respectivemente ['mean color'], para receber a cor media, herdada dos segmentos.
###3.1 - Atual banco de treinamento feito a partir de imagem 1300X900
Verificou-se que imagens de resoluções maiores melhoravam o acerto do weka.
###3.2 - Contabilizar o tempo gasto pela classificacao

##4 - Implementar
[Vermelho]Correcao para Problema 2.0.
[Vermelho]Verificar possibilidade de merge na segmentacao{Descricao: 
	Analisou que seria mais pratico poder juntar os segmentos celulas, assim podendo apenas percorrer
os segmentos e conta-los a partir das diferencas. Jntando nodos semelhantes adjacentes ficaria mais facil
identificar as celulas.
}.