```markdown
# Agenda Nacional NLP

## Overview
This project is designed to process and analyze text data using machine learning and natural language processing techniques for the Agenda Nacional themes. It includes functionalities for generating embeddings, calculating similarities, and categorizing text data into predefined themes.

## Requirements
The project requires the following Python packages, which are listed in the `requirements.txt` file:
- scikit-learn
- numpy
- pandas
- plotly
- matplotlib
- scipy
- torch
- transformers
- pillow
- wordcloud
- python-docx
- seaborn
- nltk

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage
### Generating Embeddings and Calculating Similarities
The script `src/embeddings/produtos.py` processes JSON files in the `dados/pre_processados/produtos/` directory, generates embeddings for the text data, calculates similarities with predefined themes, and saves the results to an Excel file.

### Saving Data to Excel
The script `src/embeddings/potencial.py` saves processed data to an Excel file `dados/potencial.xlsx`.

## Running the Scripts
To run the scripts, use the following commands:
```bash
python src/embeddings/produtos.py
python src/embeddings/potencial.py
```

## Directory Structure
```
project_root/
├── dados/
│   ├── potencial.xlsx
│   ├── produtos_com_temas.xlsx
│   └── pre_processados/
│       └── produtos/
│           └── *.json
├── requirements.txt
├── src/
│   └── embeddings/
│       ├── potencial.py
│       └── produtos.py
└── readme.md
```

## License
This project is licensed under the MIT License.
```

## Acknowledgements
- [Hugging Face](https://huggingface.co/) for the `transformers` library.
- [NLTK](https://www.nltk.org/) for natural language processing tools.
```