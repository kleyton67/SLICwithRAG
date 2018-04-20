#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:32:33 2018

@author: dragao
"""

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

from skimage import segmentation
from skimage.future import graph
from skimage.util import img_as_float
import numpy as np
import argparse
import cv2
import os
import time
from extrai_atributos.extraiAtributos import ExtraiAtributos
from wekaWrapper import Weka
import arff2svm

# Lê os parâmetros da linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-dir_base", "--diretorio_base", required=False, help="Diretorio com imagens padrao",
                default = "arquivos_base/",type=str)
ap.add_argument("-dir_comp", "--diretorio_compactado", required = False, 
                help="Diretorio com imagens compactadas\nPra nao compactar utilizar -es 0(fator dde escalonamento = 0)",
                default = "arquivos_compactados/", type=str)
ap.add_argument("-dir_res", "--diretorio_resultado", required = False, 
                help="Diretorio imagens de resultado (imagens originais + imagens classificadas)",
                default = "arquivos_resultados/", type=str)
ap.add_argument("-b", "--banco", required=False, help="Caminho do banco de imagens", default="../data/demo",
                type=str)
ap.add_argument("-mse", "--maxsegmentos", required=False, help="Número máximo de segmentos", default=1000,
                type=int)
ap.add_argument("-se", "--segmentos", required=False, help="Número aproximado de segmentos", default=400,
                type=int)
ap.add_argument("-si", "--sigma", required=False, help="Sigma", default=5, type=int)
ap.add_argument("-sc", "--compactness", required=False, help="Higher values makes superpixel shapes more square/cubic",
                default=6.0, type=float)
ap.add_argument("-so", "--outline", required=False, help="Deixa borda do superpixel mais larga: 0 ou 1.", default=0,
                type=int)
ap.add_argument("-c",  "--classname", required=False, help="Classificador", default="weka.classifiers.trees.J48",
                type=str)
ap.add_argument("-co", "--coptions", required=False, help="Opcoes do classificador", default="-C 0.3", type=str)
ap.add_argument("-es", "--escalonamento", required = False, help="Fator de redução da Imagem", default = 0.2, type=float)


args = vars(ap.parse_args())

# Atualiza os parâmetros com os valores passados na linha de comando ou com os defaults

p_diretorio = args["diretorio_base"]
p_compactado = args["diretorio_compactado"]
p_resultado = args["diretorio_resultado"]
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

#Separa o nome do banco onde está o banco de imagens
pasta_raiz, nome_banco = os.path.split(pasta_banco_imagens)
pasta_banco_imagens = pasta_banco_imagens + "/"
pasta_raiz = pasta_raiz + "/"

classeAtual = 0

def listar_imagens_diretorio(caminhos_elementos):
    """
    parametros:
        caminhos_eleentos é o diretorio
        
    retorno:
        elementos no diretorio
    """
    lista_compactados = []
    for informacao in caminhos_elementos:
        caminho, imagens = os.path.split(informacao)
        lista_compactados+=[imagens,]
        return lista_compactados


def compactar_imagens(pasta_base, pasta_compactada):
    """
        parametros:
            pasta_base  é o caminho com as imagens com tamanho original
            pasta_compactada é o caminho para as imagens compactadas
            
        retorno
            bool para especificar que todas as imagens foram com-
            pactadas com sucesso (TRUE) ou ocorreu algum erro(FALSE)
    
    """
    print"--- Compactando Elementos ---"
    lista_compactados = []
    elementos_in_compactado = []
    arquivos = [os.path.join(pasta_base, nome) for nome in os.listdir(pasta_base)]
    arquivos_compactados = [os.path.join(pasta_compactada, nome) for nome in os.listdir(pasta_compactada)]
    jpgs = [arq for arq in arquivos if arq.lower().endswith(".jpg")]
    jpgs_compactados = [arq for arq in arquivos_compactados if arq.lower().endswith(".jpg")]
    
    lista_compactados += [listar_imagens_diretorio(jpgs_compactados), ]
    
    for img in jpgs:
        caminho, nome = os.path.split(img)
        if nome not in lista_compactados:
            print "Elemento compactado %s" % nome
            image = cv2.imread(img)
            imagem_compactada = np.flipud(cv2.resize(image,None,fx=escala, fy=escala, interpolation = cv2.INTER_CUBIC))
            #guardar imagem compactada na pasta_compactada
            cv2.imwrite(pasta_compactada+nome, imagem_compactada)
    
    elementos_in_compactado += [listar_imagens_diretorio(jpgs_compactados), ]
    print"--- Fim da Compactacao de Elementos ---"
    return elementos_in_compactado

def segmentacao_slic():
    global segments
    print "Segmentos = %d, Sigma = %d, Compactness = %0.2f e Classe Atual = %d" % (p_segmentos, p_sigma, p_compactness, classeAtual)
    start_time = time.time()
    segments = segmentation.slic(img_as_float(image), n_segments=p_segmentos, sigma=p_sigma, compactness=p_compactness)+1
    print("--- Tempo Python skikit-image SLIC: %s segundos ---" % (time.time() - start_time))

cores = [(255, 255, 255), (0, 0, 0)]

    
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
    
# Atualiza a imagem pintando o superpixel com a cor da classe informada
def updateImagem(mask, mask_inv, classeAtual):
    global image, c_image
    cor_classe = np.zeros((height,width,3), np.uint8)
    cor_classe[:, :] = cores[classeAtual]
    image_classe = cv2.addWeighted(c_image, 0, cor_classe, 10, 0)
    image_classe= cv2.bitwise_and(image_classe, image_classe, mask=mask)
    
    image = cv2.bitwise_and(image, image, mask=mask_inv)
    mask[:] = 255  
    image = cv2.bitwise_or(image, image_classe, mask=mask)

# Extrai os atributos de todas as imagens e salva no arquivo training.arff
def extractArff(event = None, nomeArquivoArff = 'training.arff', nomeImagem = None, classes = None, overwrite = True):
    extraiAtributos = ExtraiAtributos(nome_banco, pasta_raiz)
    if nomeImagem is None:
        extraiAtributos.extractAll(nomeArquivoArff=nomeArquivoArff, classes=classes, overwrite=overwrite )	
    else:
        extraiAtributos.extractOneFile(nomeArquivoArff=nomeArquivoArff, nomeImagem=nomeImagem )

	pathToFile = extraiAtributos.nomePastaRaiz + extraiAtributos.nomeBancoImagens + "/"
	arff2svm.transform(pathToFile+nomeArquivoArff,pathToFile+nomeArquivoArff+".svm")
 
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

def grafo():
    '''
    
        Responsavel pelas operacoes com grafos
    
    '''
    print "\n--- Contagem e união dos segmentos ---\n"
    start_time = time.time()
    global rag, image, segments
    labels = segments+1
    #incluir caixa de seleção
    rag = graph.rag_mean_color(img_as_float(image), labels,sigma=p_sigma) 
    labels2 = graph.cut_threshold(labels, rag, 1)
    new_rag = graph.rag_mean_color(img_as_float(image), labels2,sigma=p_sigma)
    new_cel = percorrer_rag_adj_nodos(new_rag)
    segments = labels2
    rag = new_rag
    print "\n--- Fim da uniao dos segmentos ---\n"
    print("--- Tempo para união dos seguimentos --- %s segundos" % (time.time() - start_time))
    return len(new_cel)
    
#Fim da operacoes com a rag

def classify():
    global image
    
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
    
def salvar_diretorio_nome_celulas(diretorio, nome_imagem, total_celulas):
    """
        Salva no diretorio a imagem original com a imagem de classificacao com 
        o nome : nome+.numero_celulas.jpeg
        
        parametros: 
           diretorio : diretorio da imagem
           nome_imagem : nome da imagem
           total_celulas : numero de celulas encontradas pela classificacao
           
        retorno:
            NULL
    """
    global image, c_image
    print diretorio
    print nome_imagem
    print total_celulas
    nome_imagem = nome_imagem.strip(".jpg")
    if not os.path.exists(diretorio+nome_imagem):
        os.mkdir(diretorio+nome_imagem)
    cv2.imwrite(diretorio+"/"+nome_imagem+"/"+nome_imagem+"."+str(total_celulas)+".jpg", image)
    cv2.imwrite(diretorio+"/"+nome_imagem+"/"+nome_imagem+"res."+str(total_celulas)+".jpg", c_image)
    
    
    
compactar_imagens(p_diretorio, p_compactado )
dicionario = {0 : 0}
rag = 0
segments = 0

arquivos_compactados = [os.path.join(p_compactado, nome) for nome in os.listdir(p_compactado)]
jpgs_comp_completo = [arq for arq in arquivos_compactados if arq.lower().endswith(".jpg")]

print "procurar no diretorio"
print arquivos_compactados

#image = cv2.imread(jpgs_comp_completo[1])
#c_image = image.copy()
#caminho, nome = os.path.split(jpgs_comp_completo[0])
#salvar_diretorio_nome_celulas(p_resultado, nome, 50)
i=0
for img in jpgs_comp_completo:
        caminho, nome = os.path.split(img)
        print nome
        image = cv2.imread(jpgs_comp_completo[i])
        c_image = image.copy()
        height, width, channels = image.shape
        segmentacao_slic()
        classify()
        dicionario[nome] = grafo()
        print p_resultado
        print nome
        print dicionario[nome]
        salvar_diretorio_nome_celulas(p_resultado, nome, dicionario[nome])
        i+=1
        
print dicionario