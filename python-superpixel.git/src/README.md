#Projeto Identificacao e Contagem de celulas usando Weka e rag_mean_color(scikit_image)
##Colaboradores
Prof. Mt. Willian Paraguassu
Kleyton Sartori Leite

##Problemas
###Grafo direcionado?
Usando grafo direcionado seria mais rapido?(Observar o comportamento na funcao grafo)
###Total de superpixeis utilizados
O atual algoritmo nao consegui identificar celulas que exigijam uma verificacao de grau > 1
###Treinamentos perfeito para qualquer dimensao de imagem?
Verificou-se erros para treinamentos de imagens de resoluções diferentes

##Resolvido
###Acesso da informacao do nodo
Normalmente as funcoes da rag, colocam apenas o peso entre os nodos, assim nas funcoes de adjacencia
sao retornados vertices, com peso de suas arestas, porem isso pode ser corrigido utilizando o metodo
node e respectivemente ['mean color'], para receber a cor media, herdada dos segmentos.
###Atual banco de treinamento feito a partir de imagem 1300X900
Verificou-se que imagens de resoluções maiores melhoravam o acerto do weka.
###Contabilizar o tempo gasto pela classificacao
