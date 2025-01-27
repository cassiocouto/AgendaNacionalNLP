from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import math
import os
import json

import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch

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
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    outputs = model(**inputs)
    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
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
    # "Sistemas educacionais: Educação Básica (Qualidade da educação básica: formação de professores), doutores em educação nas escolas, estratégias de aprendizagem, formação de professores, gestão de políticas de educação, formação de gestão pública, tecnologias educacionais",
    "Saúde: Pesquisa e Desenvolvimento (P&D) de medicamentos, envelhecimento, modelagem epidemiológica",
    "Saúde: equidade, Sistema Único de Saúde (SUS), sistema industrial da saúde",
    "Agro e bioeconomia: agregação de valor, economia circular, adaptação a mudanças climáticas",
    "Meio ambiente e agronomia",
    "Sistemas agroalimentares: agregação de valor, segurança alimentar, produção mais sustentável, sanidade, cooperativismo",
    "Sistemas agroindustriais: agregar valor cadeia industrial, recursos humanos, meio ambiente, alimentos",
]

# Convert RS_themes to embeddings
RS_themes_embeddings = [get_embedding(theme) for theme in RS_themes]

# Define the directory path
directory_path = "dados/pre_processados/produtos/"

# List all files in the directory
files = os.listdir(directory_path)

# Filter out only JSON files
json_files = [file for file in files if file.endswith(".json")]


results = []
counter = 0
for json_file in json_files:
    # Load the JSON file
    file_path = os.path.join(directory_path, json_file)
    with open(file_path, "r", encoding="iso-8859-1") as file:
        data = json.load(file)["data"][0]

        # Extract the text from the JSON
        text = (
            data["NM_PRODUCAO"]
            + " "
            + data["DS_RESUMO"]
            + " "
            + data["DS_PALAVRA_CHAVE"]
        )

        # Calculate the similarity
        tema, confianca = calcular_similaridade(text, RS_themes_embeddings)

        # Append the results
        results.append(
            {
                "nm_area_conhecimento": data["NM_AREA_CONHECIMENTO"],
                "nm_area_avaliacao": data["NM_AREA_AVALIACAO"],
                "nm_linha_pesquisa": data["NM_LINHA_PESQUISA"],
                "nm_grande_area_conhecimento": data["NM_GRANDE_AREA_CONHECIMENTO"],
                "nm_grau_academico": data["NM_GRAU_ACADEMICO"],
                "tema": tema,
                "confianca": confianca,
            }
        )
        print(f"{counter} processed {json_file}")
        counter += 1


# Create a DataFrame from the results
produtos_df = pd.DataFrame(results)

# Save the DataFrame to an Excel file
produtos_df.to_excel("dados/produtos_com_temas.xlsx", index=False)
