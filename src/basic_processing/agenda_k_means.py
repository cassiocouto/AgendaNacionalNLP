from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

from transformers import pipeline

from src.pre_processing.read_doc import read_themes_from_docx, preprocess_text_and_tokenize, stopwords


# Criar vetores TF-IDF para os temas limpos
def vectorize_text(data):
    vectorizer = TfidfVectorizer(max_features=1000)  # Limitar a 1000 palavras mais frequentes
    X = vectorizer.fit_transform(data['cleaned_theme'])
    return X, vectorizer

data = read_themes_from_docx("data/agenda_nacional_temas.docx")
data = preprocess_text_and_tokenize(data, stopwords)
X, vectorizer = vectorize_text(data)

num_clusters = 16  # Isso pode ser ajustado conforme o tamanho do dataset

# Executar o K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Adicionar o rótulo de cluster aos dados
data['cluster'] = kmeans.labels_

fig = px.scatter(data, x='cleaned_theme', y='cluster', color='cluster', hover_data=['state'])
# fig.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())  # X.toarray() converte a matriz esparsa em densa

# Adicionar as colunas de PCA ao DataFrame
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

# Definir as cores dos clusters
colors = plt.cm.get_cmap('viridis', len(set(data['cluster'])))

# Iniciar a figura para o gráfico
plt.figure(figsize=(10, 6))

# Função para desenhar o Convex Hull de um cluster
def plot_convex_hull(points, color, label):
    hull = ConvexHull(points)
    polygon = plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], 
                       color=color, alpha=0.4, label=f'Cluster {label}', edgecolor='black', linewidth=1.5)

# Plotar os clusters com seus polígonos
for cluster_label in sorted(data['cluster'].unique()):
    # Selecionar os pontos do cluster atual
    cluster_points = data[data['cluster'] == cluster_label][['PCA1', 'PCA2']].values
    
    # Desenhar o Convex Hull (polígono) com transparência para o cluster
    plot_convex_hull(cluster_points, color=colors(cluster_label), label=cluster_label)
    
    # Plotar os pontos com opacidade mais baixa e menor destaque
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                color=colors(cluster_label), s=20, alpha=0.5, edgecolor='none')  # Tornar os pontos mais discretos

plt.title('Clusters com Tamanho Proporcional (PCA)')
plt.xlabel('Componente PCA 1')
plt.ylabel('Componente PCA 2')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Clusters')  # Legenda posicionada fora do gráfico
plt.tight_layout()  # Ajustar layout para que a legenda não sobreponha o gráfico

plt.show()

# Criar um DataFrame com os temas e os clusters
df_clusters = pd.DataFrame({
    'state': data['state'],           # Coluna com os estados
    'region': data['region'],         # Coluna com as regiões
    'tema': data['theme'],           # Coluna com os temas originais
    'cluster': data['cluster'],      # Coluna com o cluster atribuído
})

# Exportar para CSV
df_clusters.to_excel('data/clusters_temas_16.xlsx', index=False)


# Função para contar a ocorrência de temas por estado e região
def count_themes_by_group(data, group_col, theme_col='cluster'):
    # Contagem de temas por estado/região
    return data.groupby([group_col, theme_col]).size().reset_index(name='count')

# Função para recuperar os temas correspondentes aos clusters comuns
def get_common_themes(data, common_clusters):
    # Filtrar os temas que estão nos clusters comuns
    common_themes = data[data['cluster'].isin(common_clusters['cluster'])]
    return common_themes[['state', 'region', 'theme', 'cluster']].drop_duplicates()

# Contagem de temas por estado e por região
theme_count_by_state = count_themes_by_group(data, 'state')
theme_count_by_region = count_themes_by_group(data, 'region')

# Contar quantos estados por tema (cluster) e verificar a porcentagem
total_states = data['state'].nunique()
state_occurrence = theme_count_by_state.groupby('cluster')['state'].nunique().reset_index(name='state_count')
state_occurrence['percentage'] = (state_occurrence['state_count'] / total_states) * 100

# Filtrar temas comuns a pelo menos 80% dos estados
common_themes_national = state_occurrence[state_occurrence['percentage'] >= 80]

# Contar quantas regiões por tema (cluster)
region_occurrence = theme_count_by_region.groupby('cluster')['region'].nunique().reset_index(name='region_count')
total_regions = data['region'].nunique()

# Verificar temas comuns em todas as regiões
common_themes_by_region = region_occurrence[region_occurrence['region_count'] == total_regions]

# Recuperar as strings dos temas comuns
common_national_themes = get_common_themes(data, common_themes_national)
common_regional_themes = get_common_themes(data, common_themes_by_region)

# Exibir os temas comuns por estado e região
print(f'Temas transversais em pelo menos 80% dos estados:\n{common_national_themes}')
print(f'Temas transversais em todas as regiões:\n{common_regional_themes}')

# Se quiser salvar os resultados
#common_national_themes.to_excel('data/temas_comuns_80_porcento_uf.xlsx', index=False)
#common_regional_themes.to_excel('data/temas_comuns_todas_regioes.xlsx', index=False)


# Função para sumarizar os temas de cada cluster
def summarize_cluster(themes, model):
    # Concatenar os temas em uma string
    text = '. '.join(themes)
    
    # Limitar o texto para evitar textos muito grandes para a sumarização
    max_tokens = 256  # Limite típico de tokens que o BART pode lidar
    if len(text.split()) > max_tokens:
        text = ' '.join(text.split()[:max_tokens]) + '...'

    # Garantir que o texto não está vazio
    if len(text.strip()) == 0:
        return "Resumo não disponível. Texto insuficiente para gerar um resumo."

    try:
        # Gerar sumarização
        summary = model(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Erro ao gerar o resumo: {str(e)}"

# Função para gerar sumarização para cada cluster
def generate_cluster_summaries(data, model):
    summaries = {}
    for cluster_label in sorted(data['cluster'].unique()):
        # Filtrar os temas que pertencem a este cluster
        cluster_themes = data[data['cluster'] == cluster_label]['theme'].tolist()
        
        # Gerar a sumarização para os temas deste cluster
        summaries[cluster_label] = summarize_cluster(cluster_themes, model)
    
    return summaries

# Função para contar a ocorrência de temas por estado e região
def count_themes_by_group(data, group_col, theme_col='cluster'):
    return data.groupby([group_col, theme_col]).size().reset_index(name='count')

# Função para recuperar os temas correspondentes aos clusters comuns
def get_common_themes(data, common_clusters):
    return data[data['cluster'].isin(common_clusters['cluster'])][['state', 'region', 'theme', 'cluster']].drop_duplicates()

# Preparar o modelo de sumarização (Hugging Face)
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

# Exemplo de uso da função de sumarização
# Primeiro, gerar as sumarizações dos clusters com base nos temas
cluster_summaries = generate_cluster_summaries(data, summarizer)

# Exibir as sumarizações geradas para cada cluster
for cluster, summary in cluster_summaries.items():
    print(f"Cluster {cluster} - Sumarização:")
    print(summary)
    print("-" * 80)

# Se quiser salvar as sumarizações em um arquivo CSV
summary_df = pd.DataFrame(list(cluster_summaries.items()), columns=['Cluster', 'Summary'])
summary_df.to_excel('data/summarized_clusters.xlsx', index=False)