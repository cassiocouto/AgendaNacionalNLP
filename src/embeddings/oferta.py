from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import math

import torch
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np



# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("portuguese"))

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
model = AutoModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


# Function to remove stopwords
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


# Function to normalize embeddings
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


# Função para gerar embeddings
def get_embedding(text):
    text = remove_stopwords(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    embeddings = sum_embeddings / sum_mask
    return embeddings.detach().numpy()


def calcular_similaridade(texto, tema_embeddings):
    theme_embedding = get_embedding(texto)
    theme_embedding = normalize_embedding(theme_embedding)
    similarities = [
        cosine_similarity(theme_embedding.reshape(1, -1), theme_emb.reshape(1, -1))[0][
            0
        ]
        for theme_emb in tema_embeddings
    ]

    most_similar = similarities.index(max(similarities))
    tema = RS_themes[most_similar]
    return tema, math.ceil(max(similarities) * 100)


RS_themes = [
    "Estrutura de sistema laboratorial: pesquisa e desenvolvimento, homologação, integração de soluções em plataforma",
    "Economia marinha: oceanos",
    "Eletrônica e bioengenharia: ecossistema do Inventário Nacional da Diversidade Linguística (INDL), transversalidade, inteligência artificial, robótica, Indústria 4.0",
    "Semicondutores: inovação colaborativa de grupos, materiais aplicação industrial, Projeto de Intercomparação de Modelo Acoplado (CMIP), nanomateriais, materiais avançados",
    "Biotecnologia: Scale-up, ambiente regulatório",
    "Incentivo às tecnologias: ciência de dados (inteligência artificial, inteligência estratégica, robótica), inovação e empreendedorismo",
    "Tecnologias sociais e empreendedorismo de impacto",
    "Mudanças climáticas e consequências: mitigação e adaptação",
    "Adaptação climática e gestão de riscos: meteorologia, mapeamento, sistema alerta, sistemas de gestão",
    "Sustentabilidade ambiental e social",
    "Energia: fontes renováveis e sustentáveis, economia circular, soluções para contingência em desastres",
    "Desenvolvimento econômico sustentável: diversificação de matrizes e energia sustentável",
    "Transição energética sustentável: Smart grid, descarbonização, hidrogênio verde, sistemas de armazenamento (baterias), cadeias industriais verdes (amônia etc.)",
    #"Sistemas educacionais: Educação Básica (Qualidade da educação básica: formação de professores), doutores em educação nas escolas, estratégias de aprendizagem, formação de professores, gestão de políticas de educação, formação de gestão pública, tecnologias educacionais",
    "Saúde: Pesquisa e Desenvolvimento (P&D) de medicamentos, envelhecimento, modelagem epidemiológica",
    "Saúde: equidade, Sistema Único de Saúde (SUS), sistema industrial da saúde",
    "Agro e bioeconomia: agregação de valor, economia circular, adaptação a mudanças climáticas",
    "Meio ambiente e agronomia",
    "Sistemas agroalimentares: agregação de valor, segurança alimentar, produção mais sustentável, sanidade, cooperativismo",
    "Sistemas agroindustriais: agregar valor cadeia industrial, recursos humanos, meio ambiente, alimentos",
]

# Convert RS_themes to embeddings
RS_themes_embeddings = [get_embedding(theme) for theme in RS_themes]


oferta = pd.read_excel(r"dados\pre_processados\oferta.xlsx")
oferta = oferta.fillna("")
for index, row in oferta.iterrows():
    '''
    texto = row['NM_PROGRAMA'].strip() + ' ' + row['NM_LINHA_PESQUISA'].strip() + ' ' + row['NM_AREA_CONCENTRACAO'].strip()
    if row["NM_LINHA_PESQUISA"] == 'PROJETO ISOLADO':
        texto = row['NM_PROGRAMA'].strip() + ' ' + row['NM_AREA_CONCENTRACAO'].strip()   
    texto = re.sub(r'[0-9 ]+[\.\-\)\_][0-9 ]*', '', texto)
    tema, confianca = calcular_similaridade(texto, RS_themes_embeddings)
    '''
    # Pesos
    peso_nm_programa = 10
    peso_nm_linha_pesquisa = 1
    peso_nm_area_concentracao = 5

    # Calcular similaridades
    tema_nm_programa, confianca_nm_programa = calcular_similaridade(row['NM_PROGRAMA'], RS_themes_embeddings)
    tema_nm_linha_pesquisa, confianca_nm_linha_pesquisa = calcular_similaridade(row['NM_LINHA_PESQUISA'], RS_themes_embeddings)
    tema_nm_area_concentracao, confianca_nm_area_concentracao = calcular_similaridade(row['NM_AREA_CONCENTRACAO'], RS_themes_embeddings)

    # Calcular média ponderada das confianças
    confianca_total = (confianca_nm_programa * peso_nm_programa + 
                    confianca_nm_linha_pesquisa * peso_nm_linha_pesquisa + 
                    confianca_nm_area_concentracao * peso_nm_area_concentracao) / (peso_nm_programa + peso_nm_linha_pesquisa + peso_nm_area_concentracao)

    # Decidir o tema com maior confiança
    temas = [tema_nm_programa, tema_nm_linha_pesquisa, tema_nm_area_concentracao]
    confiancas = [confianca_nm_programa, confianca_nm_linha_pesquisa, confianca_nm_area_concentracao]
    tema_final = temas[confiancas.index(max(confiancas))]

    # Resultado final
    tema = tema_final
    confianca = confianca_total

    oferta.at[index, 'TEMA'] = tema
    oferta.at[index, 'CONFIANCA'] = confianca

oferta.to_excel(r"dados\oferta_com_temas.xlsx", index=False)
