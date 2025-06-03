# TP_IACD_Pordata

Este projeto tem como objetivo recolher, analisar e visualizar dados estatísticos do site [PORDATA](https://www.pordata.pt/pt/estatisticas). A partir dos dados obtidos, o utilizador poderá gerar gráficos que facilitam a interpretação e análise de tendências.

## Funcionalidade

O programa permite:

- Inserir **links da PORDATA** correspondentes a temas estatísticos de interesse;
- Extrair e processar os dados dessas páginas;
- Gerar automaticamente **gráficos** para análise visual dos dados.

## Exemplos de links suportados

Aqui estão alguns exemplos de páginas da PORDATA compatíveis com o programa:

- [Empresas por dimensão](https://www.pordata.pt/pt/estatisticas/empresas/caracterizacao-e-demografia/empresas-por-dimensao)
- [Ganho médio mensal](https://www.pordata.pt/pt/estatisticas/salarios-e-pensoes/salarios/ganho-medio-mensal)
- [Desemprego registado por tempo](https://www.pordata.pt/pt/estatisticas/emprego/populacao-desempregada/desemprego-registado-nos-centros-de-emprego-por-tempo)

---

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/davgomes92/TP-IACD-Pordata.git
```

### 2. Mudar para o diretório do projeto

```bash
cd TP-IACD-Pordata
```
```markdown
> Nota: O diretório criado será, por padrão, `TP-IACD-Pordata`, a menos que tenha especificado outro nome ao clonar.
```

### 3. Instalar as dependências

Recomenda-se o uso de um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Executar o script principal

```bash
python ficheiro_execucao.py
```

> O programa irá solicitar os links da PORDATA que pretende analisar.


## Estrutura do Projeto

```
TP-IACD-Pordata/
├── ficheiro_execucao.py
├── requirements.txt
├── /src
│   └── ... (módulos auxiliares)
└── README.md
```

## Requisitos

- Python 3.7 ou superior
- Ligação à internet (para recolha dos dados da PORDATA)

## Contacto

Para dúvidas ou sugestões, por favor contacte [@davgomes92](https://github.com/davgomes92) através do GitHub.
