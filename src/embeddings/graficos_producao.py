import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import numpy as np

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
    "Sistemas educacionais: Educação Básica (Qualidade da educação básica: formação de professores), doutores em educação nas escolas, estratégias de aprendizagem, formação de professores, gestão de políticas de educação, formação de gestão pública, tecnologias educacionais",
    "Saúde: Pesquisa e Desenvolvimento (P&D) de medicamentos, envelhecimento, modelagem epidemiológica",
    "Saúde: equidade, Sistema Único de Saúde (SUS), sistema industrial da saúde",
    "Agro e bioeconomia: agregação de valor, economia circular, adaptação a mudanças climáticas",
    "Meio ambiente e agronomia",
    "Sistemas agroalimentares: agregação de valor, segurança alimentar, produção mais sustentável, sanidade, cooperativismo",
    "Sistemas agroindustriais: agregar valor cadeia industrial, recursos humanos, meio ambiente, alimentos",
]

clusters = {
    "Ciência, Tecnologia e Inovação": [
        "Estrutura de sistema laboratorial: pesquisa e desenvolvimento, homologação, integração de soluções em plataforma",
        "Eletrônica e bioengenharia: ecossistema do Inventário Nacional da Diversidade Linguística (INDL), transversalidade, inteligência artificial, robótica, Indústria 4.0",
        "Semicondutores: inovação colaborativa de grupos, materiais aplicação industrial, Projeto de Intercomparação de Modelo Acoplado (CMIP), nanomateriais, materiais avançados",
        "Incentivo às tecnologias: ciência de dados (inteligência artificial, inteligência estratégica, robótica), inovação e empreendedorismo",
        "Biotecnologia: Scale-up, ambiente regulatório",
    ],
    "Sustentabilidade e Meio Ambiente": [
        "Economia marinha: oceanos",
        "Mudanças climáticas e consequências: mitigação e adaptação",
        "Adaptação climática e gestão de riscos: meteorologia, mapeamento, sistema alerta, sistemas de gestão",
        "Sustentabilidade ambiental e social",
        "Energia: fontes renováveis e sustentáveis, economia circular, soluções para contingência em desastres",
        "Transição energética sustentável: Smart grid, descarbonização, hidrogênio verde, sistemas de armazenamento (baterias), cadeias industriais verdes (amônia etc.)",
        "Desenvolvimento econômico sustentável: diversificação de matrizes e energia sustentável",
    ],
    "Educação, Saúde e Desenvolvimento Social": [
        "Sistemas educacionais: Educação Básica (Qualidade da educação básica: formação de professores), doutores em educação nas escolas, estratégias de aprendizagem, formação de professores, gestão de políticas de educação, formação de gestão pública, tecnologias educacionais",
        "Saúde: Pesquisa e Desenvolvimento (P&D) de medicamentos, envelhecimento, modelagem epidemiológica",
        "Saúde: equidade, Sistema Único de Saúde (SUS), sistema industrial da saúde",
        "Tecnologias sociais e empreendedorismo de impacto",
    ],
    "Agro e Bioeconomia": [
        "Agro e bioeconomia: agregação de valor, economia circular, adaptação a mudanças climáticas",
        "Meio ambiente e agronomia",
        "Sistemas agroalimentares: agregação de valor, segurança alimentar, produção mais sustentável, sanidade, cooperativismo",
        "Sistemas agroindustriais: agregar valor cadeia industrial, recursos humanos, meio ambiente, alimentos",
    ],
}

colors_clusters = {
    "Ciência, Tecnologia e Inovação": "red",
    "Sustentabilidade e Meio Ambiente": "green",
    "Educação, Saúde e Desenvolvimento Social": "blue",
    "Agro e Bioeconomia": "yellow",
}


def gerar_grafico_quantidade_temas(
    df, nome_da_coluna, nome_da_coluna_no_titulo, nome_do_arquivo
):
    contagens = df[nome_da_coluna].value_counts()

    # Quebrar strings de temas muito grandes e desambiguar temas semelhantes
    def format_tema(text):
        parts = text.split(':')
        if len(parts) > 1:
            prefix = parts[0]
            suffix = parts[1].split(',')[0].strip()  # Pega a primeira parte após o dois pontos
            return f"{prefix} ({suffix})"
        return text

    contagens.index = [textwrap.fill(format_tema(text), width=50) for text in contagens.index]

    # Normalize the counts to get color intensity
    norm = plt.Normalize(contagens.min(), contagens.max())
    bar_colors = []
    for tema in contagens.index:
        for cluster, temas in clusters.items():
            if any(tema.startswith(t.split(':')[0]) for t in temas):
                bar_colors.append(colors_clusters[cluster])
                break

    fig, ax = plt.subplots(figsize=(15, 10))
    contagens.plot(kind="barh", ax=ax, color=bar_colors, width=0.8)
    plt.title(f"{nome_da_coluna_no_titulo}")
    plt.subplots_adjust(
        left=0.3, right=0.95, top=0.95, bottom=0.1
    )  # Ajuste os valores conforme necessário
    ax.set_yticklabels(
        ax.get_yticklabels(), fontsize=10
    )  # Ajuste o tamanho da fonte conforme necessário
    # salvar em arquivo png
    plt.savefig(f"graficos/{nome_do_arquivo}.png")
    plt.close(fig)




def contar_temas(df, nome_da_coluna):
    contagens = df[nome_da_coluna].value_counts().sort_values(ascending=True)
    for tema, contagem in contagens.items():
        print(f"{tema};{contagem}")

#########################
# Produção: cor verde
#########################

# Passo 1: Carregar os dados
df = pd.read_excel("dados/produtos_com_temas.xlsx").drop_duplicates()
print(f"Quantidade de produtos no RS: {len(df)}")

# Passo 2: Gerar gráficos
gerar_grafico_quantidade_temas(df, "tema", "Quantidade de trabalhos de conclusão por tema", "quantidade_temas_producao")
contar_temas(df, "tema")

# Passo 3: Gerar gráfico de quantidade de produtos por nm_grande_area_conhecimento
print("----AREA CONHECIMENTO AREA CONHECIMENTO AREA CONHECIMENTO---")	
for area_conhecimento in df["nm_grande_area_conhecimento"].unique():
    gerar_grafico_quantidade_temas(
        df[df["nm_grande_area_conhecimento"] == area_conhecimento],
        "tema",
        f"Quantidade de trabalhos de conclusão - área conhecimento: {area_conhecimento} - por tema",
        f"quantidade_temas_producao_{area_conhecimento}",
    )
    # vou aproveitar e contar os temas
    print("##################################################")
    print(f"Contagem de temas para a oferta ({area_conhecimento}):")
    contar_temas(df[df["nm_grande_area_conhecimento"] == area_conhecimento], "tema")

# Passo 4: Gerar gráficos de quantidade de produtos por nm_grau_academico
print("----GRAU ACADEMICO GRAU ACADEMICO GRAU ACADEMICO---")
for grau_academico in df["nm_grau_academico"].unique():
    gerar_grafico_quantidade_temas(
        df[df["nm_grau_academico"] == grau_academico],
        "tema",
        f"Quantidade de trabalhos de conclusão - grau acadêmico: {grau_academico} - por tema",
        f"quantidade_temas_producao_{grau_academico}",
    )
    # vou aproveitar e contar os temas
    print("##################################################")
    print(f"Contagem de temas para a oferta ({grau_academico}):")
    contar_temas(df[df["nm_grau_academico"] == grau_academico], "tema")
