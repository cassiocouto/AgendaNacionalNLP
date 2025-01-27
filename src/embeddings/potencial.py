import xml.etree.ElementTree as ET
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math 
import os
import nltk
from nltk.corpus import stopwords
import torch
import numpy as np

# Carregar o modelo de embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))


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
RS_themes_embeddings = [normalize_embedding(get_embedding(theme)) for theme in RS_themes]

# Definir o ano atual
ano_atual = datetime.now().year
limite_anos = 10  # Trabalhos realizados nos últimos 10 anos

#Funcao para extrair nome completo e orcid-id
def extrair_nome_e_orcid(root):
    nome = root.find(".//DADOS-GERAIS").get("NOME-COMPLETO")
    nome_em_citacoes = root.find(".//DADOS-GERAIS").get("NOME-EM-CITACOES-BIBLIOGRAFICAS")
    orcid_id = root.find(".//DADOS-GERAIS").get("ORCID-ID")
    return nome, nome_em_citacoes, orcid_id

# Função para extrair formação acadêmica
def extrair_formacao(root):
    formacao = []
    for formacao_node in root.findall(".//FORMACAO-ACADEMICA-TITULACAO"):
        for item in formacao_node:
            dados = {
                "nivel": item.get("NIVEL"),
                "instituicao": item.get("NOME-INSTITUICAO"),
                "curso": item.get("NOME-CURSO"),
                "ano_inicio": item.get("ANO-DE-INICIO"),
                "ano_conclusao": item.get("ANO-DE-CONCLUSAO"),
                "titulo_trabalho": item.get("TITULO-DA-DISSERTACAO-TESE"),
            }
            formacao.append(dados)
    return formacao

# Função para extrair projetos desenvolvidos nos últimos 10 anos
def extrair_projetos(root, limite_anos, ano_atual):
    projetos = []
    for projeto_node in root.findall(".//PARTICIPACAO-EM-PROJETO/PROJETO-DE-PESQUISA"):
        ano_fim = projeto_node.get("ANO-FIM")
        if ano_fim and (ano_atual - int(ano_fim)) <= limite_anos:
            dados = {
                "nome_projeto": projeto_node.get("NOME-DO-PROJETO"),
                "ano_inicio": projeto_node.get("ANO-INICIO"),
                "ano_fim": ano_fim,
                "situacao": projeto_node.get("SITUACAO"),
                "descricao": projeto_node.findtext("DESCRICAO-DO-PROJETO"),
            }
            projetos.append(dados)
    return projetos

# Função para extrair produção intelectual com detalhes adicionais
def extrair_producao_intelectual(root, limite_anos, ano_atual):
    producao = []

    # Produção bibliográfica -> trabalho em eventos
    for prod_biblio_node in root.findall(".//PRODUCAO-BIBLIOGRAFICA//TRABALHO-EM-EVENTOS"):
        for prod_biblio_item in prod_biblio_node:
            ano = prod_biblio_item.get("ANO-DO-TRABALHO")
            if ano and (ano_atual - int(ano)) <= limite_anos:
                dados = {
                    "tipo": "Trabalho em Eventos",
                    "titulo": prod_biblio_item.get("TITULO-DO-TRABALHO"),
                    "ano": ano,
                    "evento": prod_biblio_item.get("NOME-DO-EVENTO"),
                    "pais": prod_biblio_item.get("PAIS"),
                    "idioma": prod_biblio_item.get("IDIOMA"),
                    "doi": prod_biblio_item.get("DOI"),
                }
                producao.append(dados)

    # Produção bibliográfica -> artigos publicados
    for prod_biblio_node in root.findall(".//PRODUCAO-BIBLIOGRAFICA//ARTIGO-PUBLICADO"):
        for prod_biblio_item in prod_biblio_node:
            ano = prod_biblio_item.get("ANO-DO-ARTIGO")
            if ano and (ano_atual - int(ano)) <= limite_anos:
                dados = {
                    "tipo": "Artigo Publicado",
                    "titulo": prod_biblio_item.get("TITULO-DO-ARTIGO"),
                    "ano": ano,
                    "periodico": prod_biblio_item.get("TITULO-DO-PERIODICO-OU-REVISTA"),
                    "doi": prod_biblio_item.get("DOI"),
                }
                producao.append(dados)

    # Produção bibliográfica -> livro e capítulos
    for prod_biblio_node in root.findall(".//PRODUCAO-BIBLIOGRAFICA//LIVRO-E-CAPITULOS"):
        for prod_biblio_item in prod_biblio_node:
            ano = prod_biblio_item.get("ANO")
            if ano and (ano_atual - int(ano)) <= limite_anos:
                dados = {
                    "tipo": "Livro ou Capítulo",
                    "titulo": prod_biblio_item.get("TITULO"),
                    "ano": ano,
                    "tipo_producao": prod_biblio_item.get("TIPO"),
                    "doi": prod_biblio_item.get("DOI"),
                }
                producao.append(dados)
    
    # Produção bibliográfica -> textos em jornais ou revistas
    for prod_biblio_node in root.findall(".//PRODUCAO-BIBLIOGRAFICA//TEXTO-EM-JORNAL-OU-REVISTA"):
        for prod_biblio_item in prod_biblio_node:
            ano = prod_biblio_item.get("ANO")
            if ano and (ano_atual - int(ano)) <= limite_anos:
                dados = {
                    "tipo": "Texto em Jornal ou Revista",
                    "titulo": prod_biblio_item.get("TITULO"),
                    "ano": ano,
                    "tipo_producao": prod_biblio_item.get("TIPO"),
                    "doi": prod_biblio_item.get("DOI"),
                }
                producao.append(dados)
    
    # Produção bibliográfica -> demais tipos de produção bibliográfica
    for prod_biblio_node in root.findall(".//PRODUCAO-BIBLIOGRAFICA//DEMAIS-TIPOS-DE-PRODUCAO-BIBLIOGRAFICA"):
        for prod_biblio_item in prod_biblio_node:
            ano = prod_biblio_item.get("ANO")
            if ano and (ano_atual - int(ano)) <= limite_anos:
                dados = {
                    "tipo": "Demais tipos de produção bibliográfica",
                    "titulo": prod_biblio_item.get("TITULO"),
                    "ano": ano,
                    "tipo_producao": prod_biblio_item.get("TIPO"),
                    "doi": prod_biblio_item.get("DOI"),
                }
                producao.append(dados)

    # Produção bibliográfica -> artigos aceitos para publicação
    for prod_biblio_node in root.findall(".//PRODUCAO-BIBLIOGRAFICA//ARTIGO-ACEITO-PARA-PUBLICACAO"):
        for prod_biblio_item in prod_biblio_node:
            ano = prod_biblio_item.get("ANO-DO-ARTIGO")
            if ano and (ano_atual - int(ano)) <= limite_anos:
                dados = {
                    "tipo": "Artigo Aceito para Publicação",
                    "titulo": prod_biblio_item.get("TITULO-DO-ARTIGO"),
                    "ano": ano,
                    "periodico": prod_biblio_item.get("TITULO-DO-PERIODICO-OU-REVISTA"),
                    "doi": prod_biblio_item.get("DOI"),
                }
                producao.append(dados)

    return producao

# Define the directory path
directory_path = 'dados/pre_processados/curriculos/'

# List all files in the directory
files = os.listdir(directory_path)

# Filter out only XML files
xml_files = [file for file in files if file.endswith('.xml')]

# pega os dados ja processados
dados_ja_processados = 'dados/dados_com_educacao/potencial_com_temas.xlsx'
df_ja_processados = pd.read_excel(dados_ja_processados)

# Print the list of XML files
results=[]
for xml_file in xml_files:
    xml_file_path = os.path.join(directory_path, xml_file)

    # Carregar e parsear o arquivo XML
    try:
        tree = ET.parse(xml_file_path, ET.XMLParser(encoding='ISO-8859-1'))
        root = tree.getroot()

        # Extrair informações
        nome, nome_em_citacoes, orcid = extrair_nome_e_orcid(root)
        # pegar o tema de formacao academica, projetos desenvolvidos e producao intelectual dos dados já processados baseados no nome e no orcid
        tema_formacao_academica_pre = df_ja_processados.loc[df_ja_processados['nome'] == nome]['tema_formacao_academica'].values[0]
        confianca_formacao_academica_pre = float(df_ja_processados.loc[df_ja_processados['nome'] == nome]['confianca_formacao_academica'].values[0])
        tema_projetos_desenvolvidos_pre = df_ja_processados.loc[df_ja_processados['nome'] == nome]['tema_projetos_desenvolvidos'].values[0]
        confianca_projetos_desenvolvidos_pre = float(df_ja_processados.loc[df_ja_processados['nome'] == nome]['confianca_projetos_desenvolvidos'].values[0])
        tema_producao_intelectual_pre = df_ja_processados.loc[df_ja_processados['nome'] == nome]['tema_producao_intelectual'].values[0]
        confianca_producao_intelectual_pre = float(df_ja_processados.loc[df_ja_processados['nome'] == nome]['confianca_producao_intelectual'].values[0])

        # verifica se qualquer um desses temas é igual a "Sistemas educacionais: Educação Básica (Qualidade da educação básica: formação de professores), doutores em educação nas escolas, estratégias de aprendizagem, formação de professores, gestão de políticas de educação, formação de gestão pública, tecnologias educacionais"
        # se sim, entao extrai a formacao academica, projetos desenvolvidos e producao intelectual
        # e reclassifica o tema de formacao academica, projetos desenvolvidos e producao intelectual
        # se não, copia a linha do arquivo já processado pro arquivo que vai ser gerado
        if "Sistemas educacionais: Educação Básica (Qualidade da educação básica: formação de professores), doutores em educação nas escolas, estratégias de aprendizagem, formação de professores, gestão de políticas de educação, formação de gestão pública, tecnologias educacionais" in (tema_formacao_academica_pre, tema_projetos_desenvolvidos_pre, tema_producao_intelectual_pre):
            #import pdb; pdb.set_trace()
            formacao_academica = extrair_formacao(root)
            projetos_desenvolvidos = extrair_projetos(root, limite_anos, ano_atual)
            producao_intelectual = extrair_producao_intelectual(root, limite_anos, ano_atual)

        # Exibir resultados
            buffer_formacao_academica = []
            for formacao in formacao_academica:
                tema, confianca = calcular_similaridade(f"{formacao['curso']} {formacao['instituicao']}  {formacao['titulo_trabalho']}", RS_themes_embeddings)
                buffer_formacao_academica.append((tema, confianca))

            tema_formacao_academica = ""
            confianca_formacao_academica = 0
            df_formacao_academica = pd.DataFrame({
                "tema": [tema for tema, _ in buffer_formacao_academica],
                "confianca": [confianca for _, confianca in buffer_formacao_academica]
            })

            if not df_formacao_academica.empty:
                grouped_df_formacao_academica = df_formacao_academica.groupby("tema", as_index=False)['confianca'].sum()
                aux_tema_formacao_academica = grouped_df_formacao_academica.loc[grouped_df_formacao_academica['confianca'].idxmax()]
                tema_formacao_academica = aux_tema_formacao_academica['tema']
                denominator = len([tema for tema, _ in buffer_formacao_academica if tema == tema_formacao_academica])
                confianca_formacao_academica = int(aux_tema_formacao_academica['confianca']) / denominator

            buffer_projetos_desenvolvidos = []
            for projeto in projetos_desenvolvidos:
                tema, confianca = calcular_similaridade(f"{projeto['nome_projeto']} {projeto['descricao']}", RS_themes_embeddings) 
                buffer_projetos_desenvolvidos.append((tema, confianca))

            tema_projetos_desenvolvidos = ""
            confianca_projetos_desenvolvidos = 0
            df_projetos_desenvolvidos = pd.DataFrame({
                "tema": [tema for tema, _ in buffer_projetos_desenvolvidos],
                "confianca": [confianca for _, confianca in buffer_projetos_desenvolvidos]
            })

            if not df_projetos_desenvolvidos.empty:
                grouped_df_projetos_desenvolvidos = df_projetos_desenvolvidos.groupby("tema", as_index=False)['confianca'].sum()
                aux_tema_projetos_desenvolvidos = grouped_df_projetos_desenvolvidos.loc[grouped_df_projetos_desenvolvidos['confianca'].idxmax()]
                tema_projetos_desenvolvidos = aux_tema_projetos_desenvolvidos['tema']
                denominator = len([tema for tema, _ in buffer_projetos_desenvolvidos if tema == tema_projetos_desenvolvidos] )
                confianca_projetos_desenvolvidos = int(aux_tema_projetos_desenvolvidos['confianca']) / denominator

            buffer_producao_intelectual = []
            for prod in producao_intelectual:
                aux_producao_intelectual = ""
                if prod['tipo'] == 'Trabalho em Eventos':
                    aux_producao_intelectual += f"{prod['titulo']} {prod['evento']} "
                elif prod['tipo'] == 'Artigo Publicado':
                    aux_producao_intelectual += f"{prod['titulo']} {prod['periodico']} "
                elif prod['tipo'] == 'Livro ou Capítulo':
                    aux_producao_intelectual += f"{prod['titulo']} "
                elif prod['tipo'] == 'Texto em Jornal ou Revista':
                    aux_producao_intelectual += f"{prod['titulo']} "
                elif prod['tipo'] == 'Demais tipos de produção bibliográfica':
                    aux_producao_intelectual += f"{prod['titulo']} "
                elif prod['tipo'] == 'Artigo Aceito para Publicação':
                    aux_producao_intelectual += f"{prod['titulo']} {prod['periodico']} "

                tema, confianca = calcular_similaridade(aux_producao_intelectual, RS_themes_embeddings)
                buffer_producao_intelectual.append((tema, confianca))

            tema_producao_intelectual = ""
            confianca_producao_intelectual = 0
            df_producao_intelectual = pd.DataFrame({
                "tema": [tema for tema, _ in buffer_producao_intelectual],
                "confianca": [confianca for _, confianca in buffer_producao_intelectual]
            })

            if not df_producao_intelectual.empty:
                grouped_df_producao_intelectual = df_producao_intelectual.groupby("tema", as_index=False)['confianca'].sum()
                aux_tema_producao_intelectual = grouped_df_producao_intelectual.loc[grouped_df_producao_intelectual['confianca'].idxmax()]
                tema_producao_intelectual = aux_tema_producao_intelectual['tema']
                denominator = len([tema for tema, _ in buffer_producao_intelectual if tema == tema_producao_intelectual])
                confianca_producao_intelectual = int(aux_tema_producao_intelectual['confianca']) / denominator

            result = {
                "nome": nome,
                "nome_em_citacoes": nome_em_citacoes,
                "orcid": orcid,
                "tema_formacao_academica": tema_formacao_academica,
                "confianca_formacao_academica": confianca_formacao_academica,
                "tema_projetos_desenvolvidos": tema_projetos_desenvolvidos,
                "confianca_projetos_desenvolvidos": confianca_projetos_desenvolvidos,
                "tema_producao_intelectual": tema_producao_intelectual,
                "confianca_producao_intelectual": confianca_producao_intelectual,
            }
            print(result)
            results.append(result)
        else:
            result = {
                "nome": nome,
                "nome_em_citacoes": nome_em_citacoes,
                "orcid": orcid,
                "tema_formacao_academica": tema_formacao_academica_pre,
                "confianca_formacao_academica": confianca_formacao_academica_pre,
                "tema_projetos_desenvolvidos": tema_projetos_desenvolvidos_pre,
                "confianca_projetos_desenvolvidos": confianca_projetos_desenvolvidos_pre,
                "tema_producao_intelectual": tema_producao_intelectual_pre,
                "confianca_producao_intelectual": confianca_producao_intelectual_pre,
            }
            print(result)
            results.append(result)
    except Exception as e:
        print(f"Erro ao processar {xml_file}: {e}")

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(results)
df.drop_duplicates(subset=['nome', 'orcid'], inplace=True)

# Save the DataFrame to an Excel file
df.to_excel('dados/potencial.xlsx', index=False)