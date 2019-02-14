# Desafio de qualidade de vinhos

Este repositório contém o estudo e implementação de um modelo de classificação de qualidade de vinhos.  Este projeto foi montado seguindo as premissas de [Pesquisas Reprodutíveis](https://pt.coursera.org/learn/reproducible-research), de modo que qualquer pessoa consiga chegar aos mesmos resultados que eu utilizando os passos que segui no Jupyter Notebook.

## Dependências do projeto

Todas as dependências podem ser encontradas no arquivo `requirements.txt` e abaixo estão listadas:

* Numpy
* Scikit-learn (sk-learn)
* Pandas 
* Jupyter Notebook
* Matplotlib

Para instalar as dependências execute na pasta raiz do projeto: `pip install -r requirements.txt`. Para acessar o Jupyter Notebook que criei, execute na pasta raiz do projeto `jupyter notebook`. Logo em seguida seu browser será aberto e basta selecionar o arquivo "Projeto - Qualidade de Vinhos.ipynb". 

## Estrutura do projeto

```sh
.
├── data
│   ├── __init__.py
├── helper.py
├── LICENSE
├── Projeto - Qualidade de Vinhos.ipynb
├── README.md
├── report
│   ├── html
│   │   └── Projeto - Qualidade de Vinhos.html
│   └── markdown
│       ├── output_10_0.png
│       ├── output_15_0.png
│       ├── output_28_0.png
│       ├── output_40_0.png
│       ├── output_40_1.png
│       ├── output_42_0.png
│       ├── output_45_0.png
│       ├── output_46_1.png
│       ├── output_51_0.png
│       ├── output_54_0.png
│       ├── output_8_0.png
│       └── Projeto - Qualidade de Vinhos.md
├── requirements.txt
├── tmcm_feature_engineering.py

```

A pasta report contém um arquivo html com uma versão do relatório gerado a partir do estudo feito nesse projeto. Esse arquivo contém **todos os insights e estudos feitos, bem como uma descrição detalhada de como foi elaborado o projeto**. É importante frisar que os dados utilizados para este desafio **não** foram adicionados a pasta `data`.

**Todas as referências utilizadas para a criação desse projeto estão descritas no report.**