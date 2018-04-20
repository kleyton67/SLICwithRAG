#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
  Nome: slic_parametros.py
  Autor: Hemerson Pistori (pistori@ucdb.br)
         Alessandro dos Santos Ferreira ( asf2005kn@hotmail.com )
         Diogo Soares ( diogo.ec.2013@gmail.com )
         Rodrigo Gonçalves de Branco - chamada CUDA ( rodrigo.g.branco@gmail.com )
         Kleyton Sartori Leite - Sob orientacao de Willian Paraguassu
     
  Descricão: Mostrar o resultado do SLIC usando diferentes parâmetors

  Como usar:
  $ python slic_parametros.py --imagem imagem_de_teste.png --segmentos 100 --sigma 5

"""

# Importa bibliotecas necessárias

from skimage import segmentation, io #segementation: para a slic 
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage.util import img_as_float
import numpy as np
import argparse
import cv2
import os
import time
import pylab
from matplotlib.widgets import Slider, Button
from extrai_atributos.extraiAtributos import ExtraiAtributos
from wekaWrapper import Weka

import arff2svm

# Lê os parâmetros da linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagem", required=False, help="Arquivo com a imagem", default="arquivos_base/48h1.1-000020.jpg",
                type=str)
ap.add_argument("-b", "--banco", required=False, help="Caminho do banco de imagens", default="../data/demo",    
                type=str)
ap.add_argument("-mse", "--maxsegmentos", required=False, help="Número máximo de segmentos", default=1000, type=int)
ap.add_argument("-se", "--segmentos", required=False, help="Número aproximado de segmentos", default=400, type=int)
ap.add_argument("-si", "--sigma", required=False, help="Sigma", default=5, type=int)
ap.add_argument("-sc", "--compactness", required=False, help="Higher values makes superpixel shapes more square/cubic", default=6.0,
                type=float)
ap.add_argument("-so", "--outline", required=False, help="Deixa borda do superpixel mais larga: 0 ou 1.", default=0, type=int)
ap.add_argument("-c",  "--classname", required=False, help="Classificador", default="weka.classifiers.trees.J48", type=str)
ap.add_argument("-co", "--coptions", required=False, help="Opcoes do classificador", default="-C 0.3", type=str)
ap.add_argument("-es", "--escalonamento", required = False, help="Fator de redução da Imagem", default = 0.2, type=float)

args = vars(ap.parse_args())

# Atualiza os parâmetros com os valores passados na linha de comando ou com os defaults

p_maxsegmentos = args["maxsegmentos"]
p_segmentos = args["segmentos"]
p_sigma = args["sigma"]
p_compactness = args["compactness"]
#realca a fronteira
p_outline = None if (args["outline"] == 0) else (1, 1, 0)
escala = args["escalonamento"]
classname = args["classname"]
coptions = args["coptions"].split()

pasta_banco_imagens = args["banco"]
nome_imagem_completo = args["imagem"]

rag = 0

#Redimensionando 
image = cv2.imread(nome_imagem_completo)

image = np.flipud(cv2.resize(image,None,fx=escala, fy=escala, interpolation = cv2.INTER_CUBIC))

#duplicado para guradar somente a parte pintada da rag
c_image = image.copy()
image_obj = image.copy()#somente na rag
#Separa o nome da pasta onde está a imagem do nome do arquivo dentro da pasta
caminho, nome_imagem = os.path.split(nome_imagem_completo)
#Separa o nome do banco onde está o banco de imagens
pasta_raiz, nome_banco = os.path.split(pasta_banco_imagens)
pasta_banco_imagens = pasta_banco_imagens + "/"
pasta_raiz = pasta_raiz + "/"

# Criar janela da interface (dependendo da versão do python tem que colocar 111 como parâmetro.
ax = pylab.subplot(111)
pylab.title("Gerador de Banco de Imagens de Treinamento e Teste")

# Deixe um espaço na parte de baixo da janela para as barras de rolagem (segmentos e sigma)
pylab.subplots_adjust(bottom=0.35)

# Ajusta parâmetros para mostrar a imagem inteira
height, width, channels = image.shape
pylab.axis([0, width, 0, height])

# Classe que será gerada quando o usuário clicar em um superpixel
classeAtual = 0

#COnta a classificação, poi eh necessario para reuniao
classificacao = 0

# Extrai os superpixels e mostra na tela com contornos
print "Segmentos = %d, Sigma = %d, Compactness = %0.2f e Classe Atual = %d" % (p_segmentos, p_sigma, p_compactness, classeAtual)
start_time = time.time()
segments = segmentation.slic(img_as_float(image), n_segments=p_segmentos, sigma=p_sigma, compactness=p_compactness)+1
print("--- Tempo Python skikit-image SLIC: %s segundos ---" % (time.time() - start_time))

#imprimi imagem com segmentacao
obj = ax.imshow(segmentation.mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments, outline_color=p_outline))
#obj = ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), y, outline_color=p_outline))

# Criar as barras de rolagem para manipular o número de segmentos e o sigma (parâmetros do SLIC)
slider_segmentos = Slider(pylab.axes([0.25, 0.25, 0.65, 0.03]), 'Segmentos', 10, p_maxsegmentos, valinit=p_segmentos, valfmt='%d')
slider_sigma = Slider(pylab.axes([0.25, 0.20, 0.65, 0.03]), 'Sigma', 1, 20, valinit=p_sigma)
slider_compactness = Slider(pylab.axes([0.25, 0.15, 0.65, 0.03]), 'Compactness', 0.01, 100, valinit=p_compactness)
slider_classes = Slider(pylab.axes([0.25, 0.10, 0.65, 0.03]), 'Classe Atual', 0, 2, valinit=classeAtual, valfmt='%d')

# Cria botoes laterais
button_arff = Button(pylab.axes([0.80, 0.8, 0.15, 0.04]), 'Gerar Arff')
button_classify = Button(pylab.axes([0.80, 0.7, 0.15, 0.04]), 'Classificar')
button_cv = Button(pylab.axes([0.80, 0.6, 0.15, 0.04]), 'CrossValid')
button_imagens = Button(pylab.axes([0.80, 0.5, 0.15, 0.04]), 'Imagens')
button_reunir = Button(pylab.axes([0.80, 0.4, 0.18, 0.04]), 'reunir e contar')

# Cores usadas para preencher os superpixels com uma cor diferente para cada classe
if(slider_classes == 2):
    slider_classes -= 1
cores = [(255, 255, 255), (0, 0, 0)]

# Ponteiro para a janela
fig = pylab.gcf()


# Determina o que fazer quando os valores das barras de rolagem com os parâmetros do SLIC forem alterados
def updateParametros(val):
    global p_segmentos, p_sigma, p_compactness, segments, image

    p_segmentos = int("%d" % (slider_segmentos.val))
    p_sigma = slider_sigma.val
    p_compactness = slider_compactness.val
    
    image = c_image.copy()
    
    start_time = time.time()
    segments = segmentation.slic(img_as_float(image), n_segments=p_segmentos, sigma=p_sigma, compactness=p_compactness)+1
    print("--- Tempo Python skikit-image SLIC: %s segundos ---" % (time.time() - start_time))
    obj.set_data(segmentation.mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments, outline_color=p_outline))
    plt.draw()


# Determina o que fazer quando os valores das barras de rolagem da Classe Atual for alterado
def updateClasseAtual(val):
    global classeAtual
    classeAtual = int("%d" % (slider_classes.val))
    print "Segmentos = %d, Sigma = %d, Compactness = %0.2f e Classe Atual = %d" % (p_segmentos, p_sigma, p_compactness, classeAtual)

    
# Retorna o maior quadrado inscrito no superpixel
def largestSquare(segmento):
    h, w, c = segmento.shape
    print "height = %d, width = %d, channels = %d" % (h, w, c)
    
    S = np.zeros((h,w), np.int)
    y, x, inv = (0, 0, np.zeros((c), np.uint8))

    for i in range(1, h):
        for k in range(1, w):
            if np.array_equal(segmento[i][k][:], inv):
                S[i][k] = 0 
            else:
                S[i][k] = min(S[i][k-1], S[i-1][k], S[i-1][k-1]) + 1
            y, x = (y, x) if (S[y][x] > S[i][k]) else (i, k)

    print "y = %d, x = %d, length = %d" % (y, x, S[y][x])
            
    y, x, length = (y+1, x+1, S[y][x])
    return segmento[y-length:y, x-length:x]


# Retorna o superpixel com indice informado por parametro
def getSegmento(indice):
    mask = np.zeros(c_image.shape[:2], dtype="uint8")
    mask[segments == indice] = 255
    mask_inv = cv2.bitwise_not(mask)

    segmento = c_image.copy()
    segmento = cv2.bitwise_and(segmento, segmento, mask=mask)

    contours, _  = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

    max = -1
    maiorContorno = None
    for cnt in contours:
        if (len(cnt) > max):
            max = len(cnt)
            maiorContorno = cnt

    x,y,w,h = cv2.boundingRect(maiorContorno)
    segmento = segmento[y:y+h, x:x+w]
        
    # descomente a linha abaixo para gerar a imagem do maior quadrado dentro do superpixel
    # segmento = largestSquare(segmento)
    
    return segmento, mask, mask_inv


# Salva um arquivo .tif contendo o superpixel na pasta correspondente a classe
def saveSegmento(segmento, valorSegmento, classeAtual):
    # se aquele segmento já foi marcado antes, apaga marcação do banco de imagens
    for root, dirs, files in os.walk(pasta_banco_imagens):
        for classe in dirs:
            arquivo = pasta_banco_imagens + classe + "/" + nome_imagem + "_%05d" % valorSegmento + '.tif'
            if(os.path.isfile(arquivo)):
                os.remove(arquivo)

    pasta_classe = pasta_banco_imagens + "classe_%02d" % classeAtual
    arquivo = pasta_classe + "/" + nome_imagem + "_%05d" % valorSegmento + '.tif'
    if not os.path.exists(pasta_classe):
        os.makedirs(pasta_classe)
    cv2.imwrite(arquivo, segmento)
    

# Atualiza a imagem pintando o superpixel com a cor da classe informada
def updateImagem(mask, mask_inv, classeAtual):
    global image, c_image
    cor_classe = np.zeros((height,width,3), np.uint8)
    cor_classe[:, :] = cores[classeAtual]
    #image_classe = cv2.addWeighted(c_image, 0.7, cor_classe, 0.3, 0)
    image_classe = cv2.addWeighted(c_image, 0, cor_classe, 10, 0)
    image_classe= cv2.bitwise_and(image_classe, image_classe, mask=mask)
    
    image = cv2.bitwise_and(image, image, mask=mask_inv)
    mask[:] = 255  
    image = cv2.bitwise_or(image, image_classe, mask=mask)
    
    obj.set_data(segmentation.mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments, outline_color=p_outline))

    pylab.draw()  # Apenas associa a função que pinta o segmento (onclick) com o click do mouse


# Pinta o segmento onde o usuário clicou com a cor predefinida para a classe
# e salva o segmento como uma nova imagem na pasta correspondente à classe
def onclick(event):
    global image, c_image
    if event.xdata != None and event.ydata != None and int(event.ydata) != 0:
        x = int(event.xdata)
        y = int(event.ydata)
        print "classe atual = %d" % (classeAtual)
        print "segmento = %d" % segments[y, x]
        print "x = %d y = %d " % (x, y)        
        valorSegmento = segments[y, x]

        segmento, mask, mask_inv = getSegmento(valorSegmento)
        saveSegmento(segmento, valorSegmento, classeAtual)

        updateImagem(mask, mask_inv, classeAtual)


# Extrai os atributos de todas as imagens e salva no arquivo training.arff
def extractArff(event = None, nomeArquivoArff = 'training.arff', nomeImagem = None, classes = None, overwrite = True):
    extraiAtributos = ExtraiAtributos(nome_banco, pasta_raiz)
    if nomeImagem is None:
        extraiAtributos.extractAll(nomeArquivoArff=nomeArquivoArff, classes=classes, overwrite=overwrite )	
    else:
        extraiAtributos.extractOneFile(nomeArquivoArff=nomeArquivoArff, nomeImagem=nomeImagem )

	pathToFile = extraiAtributos.nomePastaRaiz + extraiAtributos.nomeBancoImagens + "/"
	arff2svm.transform(pathToFile+nomeArquivoArff,pathToFile+nomeArquivoArff+".svm")
 
 
#Operacoes
# Retorna o grafo desenhado na imagem
def desenho_rag(rag, labels):
    '''
        Parametro: 
            labels: array 2d
            grafo
            Imagem
        retorno:
            nada
        
        Imprime o grafo em cima da imagem
    '''
    global image
    lc = graph.show_rag(labels, rag, image)        
    
    plt.colorbar(lc, fraction=0.03)
    io.show()
    
def comparar_labels(labels, labels2):
    '''
        Imprime dois labels
    '''
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

    ax[0].imshow(labels)
    ax[1].imshow(labels2)

    for a in ax:
        a.axis('off')
    plt.show()
    

def mostrar_imagens(event):
    '''
    Parametro 
        recebe as duas imagens globais: image e c_image
    
    retorno
        Nada
        
    Plota as duas imagens uma do lado da outra na mesma janela
    '''
    global image, c_image
    
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                           figsize=(8, 4))
    
    ax[0].imshow(c_image)
    ax[0].set_title('Original')
    
    ax[1].imshow(image)
    ax[1].set_title('Apos Classificacao')
    
    for a in ax:
        a.axis('off')
    
    plt.tight_layout()
    plt.show()

def find_in_tuple_2(array, elemento):
    '''
        parametros:
            tuple:  dimensao 2
            elemento: dado procurado
            
        return:
            a, b inteiros com a posicoes do dado(sendo a primeira posicao = 0)
            -1 se elemento nao encontrado
    '''
    
    for aux in array:
        if elemento in aux:
            return array.index(aux), aux.index(elemento)
    return -1, -1

def percorrer_rag_adj_nodos(graph):
    '''
        Percorre grafo pelos nodos adjacentes  e guarda nodos diferentes do]
    fundo
        Parametros: 
            Grafo
        Return:
            Uma list de de celulas identificadas como celulas
                        
    '''    
    #celulas eh uma tupla
    celulas_totais = []
    cel = np.array([0, 0, 0])
    #n  recebe a adjacencia de um nodo
    for n in graph.adjacency():
        #n[1] eh o seu nodo, n[2] nodos adjacents em um dict
        nodo_p = n[0]
        diction = n[1]
        #comp retorna ma matriz booleana
        #n eh uma celula?
        comp = graph.node[nodo_p]['mean color']==cel
        if comp[0]:
            #n eh uma celula, mas ele ja esta no vetor?
            i, j = find_in_tuple_2(celulas_totais, nodo_p)
            if(i == -1) or (j == -1):
                celulas = [nodo_p]
            else:
                celulas = celulas_totais[i]
            for nodo, info in diction.items():
                #seu vizinho tambem eh uma celula?
                comp = graph.node[nodo]['mean color']==cel
                if comp[0]:
                    #ele ja esta no vetor?
                    if nodo not in celulas:
                        #se for celula e nao estiver no vetor adicionar
                        celulas+=(nodo,)
            celulas_totais += [celulas]

                    
    return celulas_totais

def mostrar_grafo_info(graph):
    '''
        Percorre a rag e imprime informacoes de cor media
        
    '''
    
    NDV = graph.nodes(data=True)
    for n, dd in NDV: 
        print((n, dd.get('mean color')))

def grafo(event):
    global classificacao
    '''
    
        Responsavel pelas operacoes com grafos
    
    '''
    if classificacao == 0:
        print"Executar classificação para a reunião e contagem"
        classify(1)
        
    print "\n--- Contagem e união dos segmentos ---\n"
    global rag, image, segments
    labels = segments+1
    rag = graph.rag_mean_color(img_as_float(image), labels,sigma=p_sigma) 
    labels2 = graph.cut_threshold(labels, rag, 1)
    new_rag = graph.rag_mean_color(img_as_float(image), labels2,sigma=p_sigma)
    desenho_rag(rag, labels)
    new_cel = percorrer_rag_adj_nodos(new_rag)
    segments = labels2
    desenho_rag(new_rag, labels2)
    print "Total de celulas: %d" % len(new_cel)
    rag = new_rag
    print "\n--- Fim da uniao dos segmentos ---\n"
    
#Fim da operacoes com a rag

def classify(event):
    global image, classificacao
    
    start_time = time.time()
    # Extrai os atributos e salva em um arquivo arff
    extractArff( overwrite=False )
    
    # Carrega o arquivo arff e faz o treinamento
    weka = Weka(pasta_banco_imagens)
    weka.initData('training.arff')
    weka.trainData(classname=classname, options=coptions)
    
    # Realiza a classificação para cada superpixel da imagem
    try:
	for valorSegmento in np.unique(segments):
            segmento, mask, mask_inv = getSegmento(valorSegmento)
            
            arquivo = pasta_banco_imagens + 'test.tif'
            cv2.imwrite(arquivo, segmento)
            
            extractArff(nomeArquivoArff='test.arff', nomeImagem='test.tif')
            
            classes = weka.classify('test.arff')
            
            updateImagem(mask, mask_inv, classes[0])
    finally:
        if(arquivo is not None and os.path.isfile(arquivo)):
            os.remove(arquivo)
            
    print("--- Tempo para classificacao Weka --- %s segundos" % (time.time() - start_time))
    classificacao+=1

# Realizar um teste de desempenho de classificação
def crossValidate(event):    
    # Extrai os atributos e salva em um arquivo arff
    extractArff( overwrite=False )
        
    # Realiza uma validação cruzada
    # e mostra os resultados na saída padrão
    weka = Weka(pasta_banco_imagens)
    weka.crossValidate('training.arff', classname, coptions)



fig.canvas.mpl_connect('button_press_event', onclick)

# Associa as barras de rolagens com a função update
slider_segmentos.on_changed(updateParametros)
slider_sigma.on_changed(updateParametros)
slider_compactness.on_changed(updateParametros)
button_arff.on_clicked(extractArff)
button_classify.on_clicked(classify)
button_cv.on_clicked(crossValidate)
button_imagens.on_clicked(mostrar_imagens)
button_reunir.on_clicked(grafo)


plt.show()
