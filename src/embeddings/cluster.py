import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

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
    #"Educação, Saúde e Desenvolvimento Social": [
    "Saúde e Desenvolvimento Social": [
        #"Sistemas educacionais: Educação Básica (Qualidade da educação básica: formação de professores), doutores em educação nas escolas, estratégias de aprendizagem, formação de professores, gestão de políticas de educação, formação de gestão pública, tecnologias educacionais",
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

# Preparar os dados para o DataFrame
data = []
for key, themes in clusters.items():
    for theme in themes:
        data.append([key, theme])

# Criar o DataFrame
df = pd.DataFrame(data, columns=["Cluster", "Tema"])

# Salvar o DataFrame em um arquivo Excel
df.to_excel("dados/cluster_temas_rs.xlsx", index=False)

# Vetorização dos temas usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Tema"])

# Redução de dimensionalidade usando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Adicionar as coordenadas PCA ao DataFrame
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Plotar os pontos e gerar polígonos
plt.figure(figsize=(10, 7))
colors = {
    "Ciência, Tecnologia e Inovação": "red",
    "Sustentabilidade e Meio Ambiente": "green",
    #"Educação, Saúde e Desenvolvimento Social": "blue",
    "Saúde e Desenvolvimento Social": "blue",
    "Agro e Bioeconomia": "yellow",
}

for cluster in clusters.keys():
    cluster_data = df[df["Cluster"] == cluster]
    plt.scatter(cluster_data["PCA1"], cluster_data["PCA2"], c=colors[cluster], label=cluster)
    points = cluster_data[["PCA1", "PCA2"]].values
    hull = ConvexHull(points)
    polygon = Polygon(points[hull.vertices], closed=True, alpha=0.8, color=colors[cluster])
    plt.gca().add_patch(polygon)

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.title("Rio Grande do Sul - Temas")

# Salvar o gráfico
plt.savefig("graficos/cluster.png")

# Mostrar o gráfico
plt.show()