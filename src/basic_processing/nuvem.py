import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.pre_processing.read_doc import read_docx, read_themes_from_docx, clean_stopwords, clean_ignore_phrases, stopwords, ignored_phrases
from src.pre_processing.image_processor import resize_mask

# Define Brazilian colors
brazilian_colors = ["#009C3B", "#FFDF00", "#002776"]

# Create a custom colormap
brazilian_colormap = LinearSegmentedColormap.from_list(
    "brazilian_colormap", brazilian_colors
)


def generate_wordcloud(text, image, resize=1):
    mask = np.array(Image.open(image))
    if resize > 1:
        mask = resize_mask(image, resize)
    

    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        mask=mask,
        colormap=brazilian_colormap,
    ).generate(text)
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def generate_wordcloud_per_region(data, region, image, resize =1):
    text = " ".join(data[data["region"] == region]["theme"])
    text = clean_stopwords(text, stopwords)
    generate_wordcloud(text, image, resize)


# Exemplo de texto para uma região
all_themes = read_docx("data/agenda_nacional_temas.docx")
all_themes = clean_ignore_phrases(all_themes, ignored_phrases)
all_themes = clean_stopwords(all_themes, stopwords)
generate_wordcloud(all_themes, "assets/brasil.png")

all_themes_df = read_themes_from_docx("data/agenda_nacional_temas.docx")
generate_wordcloud_per_region(all_themes_df, "NORTE", "assets/norte_modified.png")
generate_wordcloud_per_region(all_themes_df, "NORDESTE", "assets/nordeste_modified.png")
generate_wordcloud_per_region(all_themes_df, "CENTRO-OESTE", "assets/centro-oeste_modified.png")
generate_wordcloud_per_region(all_themes_df, "SUDESTE", "assets/sudeste_modified.png", 3)
generate_wordcloud_per_region(all_themes_df, "SUL", "assets/sul_modified.png", 3)

