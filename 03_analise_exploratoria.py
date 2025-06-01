import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


# Configuração do Aspeto das Vizualizações
plt.style.use('Solarize_Light2')
sns.set_palette("dark")

# Configuração dos Caminhos dos Ficheiros
caminho_entrada_base = "Dados integrados"
caminho_saida_base = "Analise exploratoria"
nome_ficheiro_entrada = "dados_integrados_pordata.csv"
caminho_completo_entrada = os.path.join(caminho_entrada_base, nome_ficheiro_entrada)

# Cria a Pasta para os ficheiros de saida se esta não exisir
os.makedirs(caminho_saida_base, exist_ok=True)

#Dicionario que agrupa os municipios em regiões
Regioes_Municipios = {
    "Entre Douro e Minho": [
        "Amares", "Arcos de Valdevez", "Barcelos", "Braga", "Caminha", "Celorico de Basto",
        "Esposende", "Fafe", "Guimarães", "Melgaço", "Monção", "Mondim de Basto",
        "Paredes de Coura", "Ponte da Barca", "Ponte de Lima", "Póvoa de Lanhoso",
        "Ribeira de Pena", "Terras de Bouro", "Valença", "Viana do Castelo",
        "Vila Nova de Cerveira", "Vila Verde", "Vizela", "Amarante", "Baião", "Felgueiras",
        "Gondomar", "Lousada", "Maia", "Marco de Canaveses", "Matosinhos", "Paços de Ferreira",
        "Paredes", "Penafiel", "Porto", "Póvoa de Varzim", "Santo Tirso", "Trofa",
        "Valongo", "Vila do Conde", "Vila Nova de Gaia", "Cinfães"
    ],
    "Trás-os-Montes e Alto Douro": [
        "Alfândega da Fé", "Alijó", "Boticas", "Bragança", "Carrazeda de Ansiães",
        "Chaves", "Freixo de Espada à Cinta", "Macedo de Cavaleiros", "Mesão Frio",
        "Miranda do Douro", "Mirandela", "Mogadouro", "Montalegre", "Murça",
        "Peso da Régua", "Resende", "Sabrosa", "Santa Marta de Penaguião",
        "Valpaços", "Vila Flor", "Vila Pouca de Aguiar", "Vila Real", "Vimioso",
        "Vinhais", "Armamar", "Lamego", "Moimenta da Beira", "Penedono",
        "São João da Pesqueira", "Sernancelhe", "Tabuaço", "Arouca", "Vila Nova de Foz Côa", "Cabeceiras de Basto", "Tarouca"
    ],
    "Beira Litoral": [
        "Águeda", "Albergaria-a-Velha", "Anadia", "Aveiro", "Espinho", "Estarreja",
        "Ílhavo", "Mealhada", "Murtosa", "Oliveira de Azeméis", "Oliveira do Bairro",
        "Ovar", "Santa Maria da Feira", "São João da Madeira", "Sever do Vouga",
        "Vale de Cambra", "Vagos", "Cantanhede", "Coimbra", "Condeixa-a-Nova",
        "Figueira da Foz", "Mira", "Montemor-o-Velho", "Penacova", "Soure",
        "Alvaiázere", "Ansião", "Batalha", "Castanheira de Pêra", "Figueiró dos Vinhos",
        "Leiria", "Marinha Grande", "Pedrógão Grande", "Pombal", "Porto de Mós",
        "Arganil", "Góis", "Lousã", "Miranda do Corvo", "Mortágua", "Oliveira do Hospital",
        "Pampilhosa da Serra", "Penela", "Tábua", "Vila Nova de Poiares"
    ],
    "Beira Interior": [
        "Aguiar da Beira", "Almeida", "Carregal do Sal", "Castro Daire", "Celorico da Beira",
        "Figueira de Castelo Rodrigo", "Fornos de Algodres", "Gouveia", "Guarda",
        "Mangualde", "Manteigas", "Mêda", "Nelas", "Oliveira de Frades", "Penalva do Castelo",
        "Pinhel", "Sabugal", "Santa Comba Dão", "São Pedro do Sul", "Seia", "Tondela",
        "Trancoso", "Viseu", "Vouzela", "Belmonte", "Castelo Branco", "Covilhã", "Fundão",
        "Idanha-a-Nova", "Oleiros", "Penamacor", "Proença-a-Nova", "Sertã", "Vila de Rei",
        "Vila Velha de Ródão", "Castelo de Paiva"
    ],
    "Estremadura e Ribatejo": [
        "Alcobaça", "Alcanena", "Alenquer", "Almeirim", "Alpiarça", "Arruda dos Vinhos",
        "Azambuja", "Benavente", "Bombarral", "Cadaval", "Caldas da Rainha", "Cartaxo",
        "Chamusca", "Constância", "Coruche", "Entroncamento", "Ferreira do Zêzere",
        "Golegã", "Lourinhã", "Nazaré", "Óbidos", "Ourém", "Peniche",
        "Rio Maior", "Salvaterra de Magos", "Santarém", "Sobral de Monte Agraço",
        "Tomar", "Torres Novas", "Torres Vedras", "Vila Nova da Barquinha"
    ],
    "Lisboa e Setúbal": [
        "Amadora", "Cascais", "Lisboa", "Loures", "Mafra", "Odivelas", "Oeiras",
        "Sintra", "Vila Franca de Xira", "Alcácer do Sal", "Alcochete", "Almada",
        "Barreiro", "Grândola", "Moita", "Montijo", "Palmela", "Santiago do Cacém",
        "Seixal", "Sesimbra", "Setúbal", "Sines"
    ],
    "Alentejo": [
        "Alandroal", "Odemira", "Alter do Chão", "Arronches", "Avis", "Borba", "Campo Maior",
        "Castelo de Vide", "Crato", "Elvas", "Estremoz", "Fronteira", "Gavião",
        "Marvão", "Monforte", "Nisa", "Ponte de Sor", "Portalegre", "Redondo",
        "Sousel", "Vila Viçosa", "Aljustrel", "Almodôvar", "Alvito", "Arraiolos",
        "Barrancos", "Beja", "Castro Verde", "Cuba", "Ferreira do Alentejo",
        "Mértola", "Moura", "Ourique", "Serpa", "Vidigueira", "Évora", "Montemor-o-Novo",
        "Mora", "Mourão", "Portel", "Reguengos de Monsaraz", "Vendas Novas", "Viana do Alentejo",

    ],
    "Algarve": [
        "Albufeira", "Alcoutim", "Aljezur", "Castro Marim", "Faro", "Lagoa", "Lagos",
        "Loulé", "Monchique", "Olhão", "Portimão", "São Brás de Alportel", "Silves",
        "Tavira", "Vila do Bispo", "Vila Real de Santo António"
    ],
    "R.A.M.": [
        "Calheta [R.A.M.]", "Câmara de Lobos", "Funchal", "Machico", "Ponta do Sol", "Porto Moniz",
        "Porto Santo", "Ribeira Brava", "Santa Cruz", "Santana", "São Vicente"
    ],
    "R.A.A.": [
        "Angra do Heroísmo", "Calheta [R.A.A.]", "Corvo", "Horta", "Lagoa [R.A.A.]", "Lajes das Flores",
        "Lajes do Pico", "Madalena", "Nordeste", "Ponta Delgada", "Povoação", "Praia da Vitória",
        "Ribeira Grande", "Santa Cruz da Graciosa", "Santa Cruz das Flores", "São Roque do Pico",
        "Velas", "Vila do Porto", "Vila Franca do Campo"
    ]
}

#Carrega o csv da integração
def carregar_dados():
    try:
        print(f"A carregar dados de: {caminho_completo_entrada}")
        df = pd.read_csv(caminho_completo_entrada, encoding='utf-8')
        print(f"Dados carregados com sucesso. Dimensões: {df.shape}")
        return df
    except Exception as erro:
        print(f"ERRO ao carregar dados: {erro}")
        return None

#Analisa informações básicas como a Dimensão do Dataset, Periodo Temporal, Numero de Municipios, Numero de anos
def informacoes_basicas(dados_df):
    print("\n" + "=" * 50)
    print("INFORMAÇÕES BÁSICAS DO DATASET")
    print("=" * 50)

    print(f"Dimensões: {dados_df.shape}")
    print(f"Período temporal: {dados_df['Ano'].min()} - {dados_df['Ano'].max()}")
    print(f"Número de Municipios únicos: {dados_df['Municipio'].nunique()}")
    print(f"Número de anos únicos: {dados_df['Ano'].nunique()}")

    print("\nTipos de dados:")
    print(dados_df.dtypes)

    print("\nPrimeiras 5 linhas:")
    print(dados_df.head())

    print("\nÚltimas 5 linhas:")
    print(dados_df.tail())

    return {
        'dimensoes': dados_df.shape,
        'periodo': (dados_df['Ano'].min(), dados_df['Ano'].max()),
        'numero_municipios': dados_df['Municipio'].nunique(),
        'numero_anos': dados_df['Ano'].nunique()
    }

# Analisa os valores em falta nas diferentes categorias do Dataset
def analisar_valores_em_falta(dados_df):
    print("\n" + "=" * 50)
    print("ANÁLISE DE VALORES EM FALTA")
    print("=" * 50)

    valores_em_falta = dados_df.isnull().sum()
    percentagem_em_falta = (valores_em_falta / len(dados_df)) * 100

    df_valores_em_falta = pd.DataFrame({
        'Valores_em_falta': valores_em_falta,
        'Percentagem': percentagem_em_falta
    })

    df_valores_em_falta = df_valores_em_falta[df_valores_em_falta['Valores_em_falta'] > 0].sort_values(
        'Valores_em_falta', ascending=False)

    if not df_valores_em_falta.empty:
        print("Colunas com valores em falta:")
        print(df_valores_em_falta)

        # Visualização dos valores em falta
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        df_valores_em_falta['Valores_em_falta'].plot(kind='bar')
        plt.title('Valores em Falta por Coluna')
        plt.xlabel('Colunas')
        plt.ylabel('Número de Valores em Falta')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        df_valores_em_falta['Percentagem'].plot(kind='bar', color='orange')
        plt.title('Percentagem de Valores em Falta')
        plt.xlabel('Colunas')
        plt.ylabel('Percentagem (%)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(caminho_saida_base, 'valores_em_falta.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        print("Não foram encontrados valores em falta no dataset.")

    return df_valores_em_falta

# Analisa as estatisticas descritivas do Dataset, neste caso:
#Contagem de valores, media, desvio padrao, minimo, maximo e os quartis 1, 2 e 3 de cada coluna numerica
#Exporta as estatisticas para um csv
def estatisticas_descritivas(dados_df):
    print("\n" + "=" * 50)
    print("ESTATÍSTICAS DESCRITIVAS")
    print("=" * 50)

    colunas_numericas = dados_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Ano' in colunas_numericas:
        colunas_numericas.remove('Ano')

    print("Colunas numéricas analisadas:", colunas_numericas)

    if colunas_numericas:
        estatisticas_desc = dados_df[colunas_numericas].describe()
        print("\nEstatísticas descritivas:")
        print(estatisticas_desc)

        estatisticas_desc.to_csv(os.path.join(caminho_saida_base, 'estatisticas_descritivas.csv'))

        return estatisticas_desc, colunas_numericas
    else:
        print("Nenhuma coluna numérica encontrada para análise.")
        return None, []

#Analisa o dataset, deteta os outliers atraves de ambos os metodos (Z-score e IQR)
#Devolve a quantidade de outliers, a que percentagem do total de valores correspondem e o boxplot correspondente
def detectar_outliers(dados_df, colunas_numericas):
    print("\n" + "=" * 50)
    print("DETECÇÃO DE OUTLIERS")
    print("=" * 50)

    informacao_outliers = {}

    for coluna in colunas_numericas:
        q1 = dados_df[coluna].quantile(0.25)
        q3 = dados_df[coluna].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        outliers_iqr = dados_df[(dados_df[coluna] < limite_inferior) | (dados_df[coluna] > limite_superior)]

        dados_limpos = dados_df[coluna].dropna()
        z_scores = np.abs(stats.zscore(dados_limpos))
        indices_outliers_zscore = dados_limpos[z_scores > 3].index
        outliers_zscore = dados_df.loc[indices_outliers_zscore]

        informacao_outliers[coluna] = {
            'contagem_iqr': len(outliers_iqr),
            'contagem_zscore': len(outliers_zscore),
            'percentagem_iqr': (len(outliers_iqr) / len(dados_df)) * 100,
            'percentagem_zscore': (len(outliers_zscore) / len(dados_df)) * 100
        }

        print(f"\n{coluna}:")
        print(f"  Outliers IQR: {len(outliers_iqr)} ({(len(outliers_iqr) / len(dados_df)) * 100:.2f}%)")
        print(f"  Outliers Z-score: {len(outliers_zscore)} ({(len(outliers_zscore) / len(dados_df)) * 100:.2f}%)")

    numero_colunas = len(colunas_numericas)
    numero_linhas = (numero_colunas + 2) // 3

    plt.figure(figsize=(15, 5 * numero_linhas))
    for indice, coluna in enumerate(colunas_numericas, 1):
        plt.subplot(numero_linhas, 3, indice)
        dados_df.boxplot(column=coluna, ax=plt.gca())
        plt.title(f'Boxplot - {coluna}')
        plt.xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(caminho_saida_base, 'boxplots_outliers.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return informacao_outliers

#Mapeia cada municipio para a sua região correspondente, com base no dicionario
def mapear_municipio_para_regiao(municipio):
    for regiao, municipios in Regioes_Municipios.items():
        if municipio in municipios:
            return regiao
    return "Região Desconhecida"

#Adiciona uma coluna de Regiao ao dataset para permitir analises por regiao
    def adicionar_coluna_regiao(dados_df):
    if 'Regiao' not in dados_df.columns:
        dados_df['Regiao'] = dados_df['Municipio'].apply(mapear_municipio_para_regiao)
    return dados_df

#Caclula correlações por par de metricas
#Devolve o valor em módulo da correlação e a matriz de correlações, identifica correlações superiores a 0.7
def matriz_correlacao(dados_df, colunas_numericas):
    print("\n" + "=" * 50)
    print("MATRIZ DE CORRELAÇÃO")
    print("=" * 50)

    if len(colunas_numericas) < 2:
        print("Número insuficiente de variáveis numéricas para análise de correlação.")
        return None

    matriz_corr = dados_df[colunas_numericas].corr()
    print("Matriz de correlação:")
    print(matriz_corr)

    plt.figure(figsize=(12, 10))
    mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))
    sns.heatmap(matriz_corr, mask=mascara, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlação entre Variáveis')
    plt.tight_layout()
    plt.savefig(os.path.join(caminho_saida_base, 'matriz_correlacao.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print("\nCorrelações fortes (|r| > 0.7):")
    correlacoes_fortes = []
    for i in range(len(matriz_corr.columns)):
        for j in range(i + 1, len(matriz_corr.columns)):
            valor_corr = matriz_corr.iloc[i, j]
            if abs(valor_corr) > 0.7:
                correlacoes_fortes.append((matriz_corr.columns[i], matriz_corr.columns[j], valor_corr))
                print(f"  {matriz_corr.columns[i]} vs {matriz_corr.columns[j]}: {valor_corr:.3f}")

    if not correlacoes_fortes:
        print("  Nenhuma correlação forte encontrada.")

    matriz_corr.to_csv(os.path.join(caminho_saida_base, 'matriz_correlacao.csv'))

    return matriz_corr, correlacoes_fortes

#Analisa as distribuições das métricas
#Devolve os histogramas
def distribuicoes_variaveis(dados_df, colunas_numericas):
    print("\n" + "=" * 50)
    print("ANÁLISE DE DISTRIBUIÇÕES")
    print("=" * 50)

    numero_colunas = len(colunas_numericas)
    numero_linhas = (numero_colunas + 2) // 3

    plt.figure(figsize=(15, 5 * numero_linhas))
    for indice, coluna in enumerate(colunas_numericas, 1):
        plt.subplot(numero_linhas, 3, indice)
        dados_df[coluna].hist(bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'Distribuição - {coluna}')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')

    plt.tight_layout()
    plt.savefig(os.path.join(caminho_saida_base, 'distribuicoes_histogramas.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Histogramas das distribuições gerados com sucesso.")

    return {}

#Mostra a evolução temporal da media dos dados
#Guarda a media dos dados por ano
def analise_temporal(dados_df):
    print("\n" + "=" * 50)
    print("ANÁLISE TEMPORAL")
    print("=" * 50)

    colunas_numericas = dados_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Ano' in colunas_numericas:
        colunas_numericas.remove('Ano')

    if not colunas_numericas:
        print("Nenhuma variável numérica disponível para análise temporal.")
        return

    dados_temporais = dados_df.groupby('Ano')[colunas_numericas].mean()

    numero_colunas = len(colunas_numericas)
    numero_linhas = (numero_colunas + 2) // 3

    plt.figure(figsize=(15, 5 * numero_linhas))
    for indice, coluna in enumerate(colunas_numericas, 1):
        plt.subplot(numero_linhas, 3, indice)
        dados_temporais[coluna].plot(kind='line', marker='o')
        plt.title(f'Evolução Temporal - {coluna}')
        plt.xlabel('Ano')
        plt.ylabel(coluna)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(caminho_saida_base, 'evolucao_temporal.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    dados_temporais.to_csv(os.path.join(caminho_saida_base, 'dados_temporais.csv'))

    return dados_temporais


#Identifica as caraterísticas redundantes, com correlação acima de limite de correlação (0.95)
def identificar_caracteristicas_redundantes(dados_df, limiar_correlacao=0.95):
    print("\n" + "=" * 50)
    print("IDENTIFICAÇÃO DE CARACTERÍSTICAS REDUNDANTES")
    print("=" * 50)

    colunas_numericas = dados_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Ano' in colunas_numericas:
        colunas_numericas.remove('Ano')

    if len(colunas_numericas) < 2:
        print("Número insuficiente de variáveis numéricas para análise de redundância.")
        return []

    matriz_corr = dados_df[colunas_numericas].corr()

    caracteristicas_redundantes = []
    for i in range(len(matriz_corr.columns)):
        for j in range(i + 1, len(matriz_corr.columns)):
            valor_corr = abs(matriz_corr.iloc[i, j])
            if valor_corr > limiar_correlacao:
                var1 = matriz_corr.columns[i]
                var2 = matriz_corr.columns[j]
                caracteristicas_redundantes.append((var1, var2, matriz_corr.iloc[i, j]))
                print(f"Características redundantes encontradas: {var1} vs {var2} (r = {matriz_corr.iloc[i, j]:.3f})")

    if not caracteristicas_redundantes:
        print(f"Nenhuma característica redundante encontrada (limiar: {limiar_correlacao}).")

    if caracteristicas_redundantes:
        n_pares = len(caracteristicas_redundantes)
        n_linhas = (n_pares + 1) // 2 

        plt.figure(figsize=(12, 5 * n_linhas))
        for idx, (var1, var2, corr) in enumerate(caracteristicas_redundantes):
            plt.subplot(n_linhas, 2, idx + 1)
            plt.scatter(dados_df[var1], dados_df[var2], alpha=0.5)
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.title(f'{var1} vs {var2}\nCorrelação: {corr:.3f}')
            if not dados_df[var1].isna().all() and not dados_df[var2].isna().all():
                z = np.polyfit(dados_df[var1].dropna(), dados_df[var2].dropna(), 1)
                p = np.poly1d(z)
                plt.plot(dados_df[var1], p(dados_df[var1]), "r--", alpha=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join('Analise exploratoria', 'caracteristicas_redundantes.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    return caracteristicas_redundantes

#Analisa o crescimento das metricas por região
#Gera gráfico de barras e um boxplot de distribuição
def analise_crescimento_por_regiao(dados_df):
    print("\n" + "=" * 50)
    print("ANÁLISE DE CRESCIMENTO POR REGIÃO")
    print("=" * 50)

    dados_df = adicionar_coluna_regiao(dados_df)
    colunas_numericas = dados_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Ano' in colunas_numericas:
        colunas_numericas.remove('Ano')

    if not colunas_numericas:
        print("Nenhuma variável numérica disponível para análise de crescimento.")
        return None

    resultados_crescimento = {}

    for coluna in colunas_numericas:
        print(f"\nAnalisando crescimento para: {coluna}")

        
        crescimento_municipios = []
        for municipio in dados_df['Municipio'].unique():
            dados_municipio = dados_df[dados_df['Municipio'] == municipio].sort_values('Ano')
            if len(dados_municipio) > 1 and not dados_municipio[coluna].isna().all():
                dados_validos = dados_municipio[dados_municipio[coluna].notna()]
                if len(dados_validos) > 1:
                    valor_inicial = dados_validos[coluna].iloc[0]
                    valor_final = dados_validos[coluna].iloc[-1]
                    if valor_inicial != 0:
                        crescimento_percentual = ((valor_final - valor_inicial) / abs(valor_inicial)) * 100
                        crescimento_municipios.append({
                            'Municipio': municipio,
                            'Regiao': dados_municipio['Regiao'].iloc[0],
                            'Valor_Inicial': valor_inicial,
                            'Valor_Final': valor_final,
                            'Crescimento_Percentual': crescimento_percentual
                        })

        if crescimento_municipios:
            df_crescimento = pd.DataFrame(crescimento_municipios)

            crescimento_por_regiao = df_crescimento.groupby('Regiao')['Crescimento_Percentual'].agg([
                'mean', 'median', 'std', 'count'
            ]).round(2)

            print(f"\nCrescimento médio por região - {coluna}:")
            print(crescimento_por_regiao.sort_values('mean', ascending=False))

            resultados_crescimento[coluna] = {
                'crescimento_municipios': df_crescimento,
                'crescimento_por_regiao': crescimento_por_regiao
            }

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            crescimento_ordenado = crescimento_por_regiao.sort_values('mean', ascending=True)
            crescimento_ordenado['mean'].plot(kind='barh', ax=ax1, color='steelblue')
            ax1.set_title(f'Crescimento Médio por Região\n{coluna}', fontsize=12)
            ax1.set_xlabel('Crescimento (%)')
            ax1.grid(True, alpha=0.3)
            
            regioes_ordenadas = crescimento_ordenado.index.tolist()
            dados_boxplot = [df_crescimento[df_crescimento['Regiao'] == regiao]['Crescimento_Percentual'].values
                             for regiao in regioes_ordenadas]

            ax2.boxplot(dados_boxplot, labels=regioes_ordenadas, vert=False)
            ax2.set_title(f'Distribuição do Crescimento por Região\n{coluna}', fontsize=12)
            ax2.set_xlabel('Crescimento (%)')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join('Analise exploratoria', f'crescimento_por_regiao_{coluna}.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
    return resultados_crescimento

#Analisa os municipios com maior crescimento por regiao
def municipios_maior_crescimento(dados_df, top_n=15):
    print("\n" + "=" * 50)
    print(f"MUNICÍPIOS COM MAIOR CRESCIMENTO (TOP {top_n})")
    print("=" * 50)

    dados_df = adicionar_coluna_regiao(dados_df)

    colunas_numericas = dados_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Ano' in colunas_numericas:
        colunas_numericas.remove('Ano')

    resultados_tabela = {}

    for coluna in colunas_numericas:
        print(f"\nTop {top_n} municípios com maior crescimento - {coluna}:")

        crescimento_municipios = []
        for municipio in dados_df['Municipio'].unique():
            dados_municipio = dados_df[dados_df['Municipio'] == municipio].sort_values('Ano')

            if len(dados_municipio) > 1 and not dados_municipio[coluna].isna().all():
                dados_validos = dados_municipio[dados_municipio[coluna].notna()]

                if len(dados_validos) > 1:
                    valor_inicial = dados_validos[coluna].iloc[0]
                    valor_final = dados_validos[coluna].iloc[-1]

                    if valor_inicial != 0:
                        crescimento_percentual = ((valor_final - valor_inicial) / abs(valor_inicial)) * 100
                        crescimento_absoluto = valor_final - valor_inicial

                        crescimento_municipios.append({
                            'Municipio': municipio,
                            'Regiao': dados_municipio['Regiao'].iloc[0],
                            'Ano_Inicial': dados_validos['Ano'].iloc[0],
                            'Ano_Final': dados_validos['Ano'].iloc[-1],
                            'Valor_Inicial': valor_inicial,
                            'Valor_Final': valor_final,
                            'Crescimento_Absoluto': crescimento_absoluto,
                            'Crescimento_Percentual': crescimento_percentual
                        })

        if crescimento_municipios:
            df_crescimento = pd.DataFrame(crescimento_municipios)

            top_crescimento = df_crescimento.nlargest(top_n, 'Crescimento_Percentual')

            print("\nTop municípios (crescimento percentual):")
            print(top_crescimento[['Municipio', 'Regiao', 'Valor_Inicial', 'Valor_Final',
                                   'Crescimento_Absoluto', 'Crescimento_Percentual']].to_string(index=False))

            print(f"\nTop 3 municípios por região - {coluna}:")
            for regiao in df_crescimento['Regiao'].unique():
                if regiao != "Região Desconhecida":
                    top_regiao = df_crescimento[df_crescimento['Regiao'] == regiao].nlargest(3,
                                                                                             'Crescimento_Percentual')
                    if not top_regiao.empty:
                        print(f"\n{regiao}:")
                        print(top_regiao[['Municipio', 'Crescimento_Percentual']].to_string(index=False))

            resultados_tabela[coluna] = {
                'todos_municipios': df_crescimento,
                'top_crescimento': top_crescimento
            }

            top_crescimento.to_csv(os.path.join('Analise exploratoria',
                                                f'top_municipios_crescimento_{coluna}.csv'),
                                                 index=False)

    return resultados_tabela

# Mostra a evolução temporal dos municipios com maior crescimento
# Municipios agrupados por regiao
def evolucao_temporal_maior_crescimento(dados_df, resultados_crescimento, top_n=10):
    print("\n" + "=" * 50)
    print("EVOLUÇÃO TEMPORAL DOS MUNICÍPIOS COM MAIOR CRESCIMENTO")
    print("=" * 50)

    dados_df = adicionar_coluna_regiao(dados_df)

    cores_regioes = {
        'Entre-Douro e Minho': '#1f77b4',
        'Trás-os-Montes e Alto Douro': '#ff7f0e',
        'Beira Litoral': '#2ca02c',
        'Beira Interior': '#d62728',
        'Estremadura e Ribatejo': '#9467bd',
        'Lisboa e Setúbal': '#8c564b',
        'Alentejo': '#e377c2',
        'Algarve': '#7f7f7f',
        'R.A.M.': '#bcbd22',
        'R.A.A.': '#17becf',
        'Região Desconhecida': '#ff0000'
    }

    for coluna, dados_crescimento in resultados_crescimento.items():
        if 'top_crescimento' in dados_crescimento:
            top_municipios = dados_crescimento['top_crescimento'].head(top_n)
            municipios_selecionados = top_municipios['Municipio'].tolist()

            print(f"\nEvoluição temporal - {coluna}:")
            print(f"Municípios analisados: {', '.join(municipios_selecionados)}")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            for municipio in municipios_selecionados:
                dados_municipio = dados_df[dados_df['Municipio'] == municipio].sort_values('Ano')
                regiao = dados_municipio['Regiao'].iloc[0]
                cor = cores_regioes.get(regiao, '#000000')

                ax1.plot(dados_municipio['Ano'], dados_municipio[coluna],
                         marker='o', label=f'{municipio} ({regiao})',
                         linewidth=2, color=cor, markersize=6)

            ax1.set_title(f'Evolução Temporal dos Top {top_n} Municípios\n{coluna}', fontsize=14)
            ax1.set_xlabel('Ano')
            ax1.set_ylabel(coluna)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

            regioes_top = top_municipios['Regiao'].value_counts()
            ax2.bar(range(len(regioes_top)), regioes_top.values,
                    color=[cores_regioes.get(reg, '#000000') for reg in regioes_top.index])
            ax2.set_title(f'Distribuição Regional dos Top {top_n} Municípios')
            ax2.set_xlabel('Região')
            ax2.set_ylabel('Número de Municípios')
            ax2.set_xticks(range(len(regioes_top)))
            ax2.set_xticklabels(regioes_top.index, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            plt.savefig(os.path.join('Analise exploratoria',
                                    f'evolucao_temporal_top_municipios_{coluna}.png'),
                                    dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

#Análise de correlação por intervalos de tempo dados
def correlacao_por_intervalos_tempo(dados_df, num_intervalos, intervalos_personalizados):
    print("\n" + "=" * 50)
    print("CORRELAÇÃO POR INTERVALOS DE TEMPO")
    print("=" * 50)

    intervalos = {}
    for i, intervalo in enumerate(intervalos_personalizados[:num_intervalos]):
        try:
            anos = intervalo.split('-')
            ano_inicio = int(anos[0])
            ano_fim = int(anos[1])
            intervalos[intervalo] = (ano_inicio, ano_fim)
        except (ValueError, IndexError):
            print(f"Formato inválido para intervalo {intervalo}. Use o formato 'ano1-ano2'.")
            continue

    if num_intervalos and num_intervalos > 0:
        intervalos = dict(list(intervalos.items())[:num_intervalos])

    colunas_numericas = dados_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Ano' in colunas_numericas:
        colunas_numericas.remove('Ano')

    if len(colunas_numericas) < 2:
        print("Número insuficiente de variáveis numéricas para análise de correlação.")
        return {}

    resultados_correlacao = {}

    for nome_intervalo, (ano_inicio, ano_fim) in intervalos.items():
        print(f"\nAnalisando período: {nome_intervalo}")

        dados_periodo = dados_df[(dados_df['Ano'] >= ano_inicio) & (dados_df['Ano'] <= ano_fim)]

        if len(dados_periodo) == 0:
            print(f"Nenhum dado disponível para o período {nome_intervalo}")
            continue

        matriz_corr = dados_periodo[colunas_numericas].corr()
        resultados_correlacao[nome_intervalo] = matriz_corr

        print(f"Número de observações: {len(dados_periodo)}")

        correlacoes_fortes = []
        for i in range(len(matriz_corr.columns)):
            for j in range(i + 1, len(matriz_corr.columns)):
                valor_corr = matriz_corr.iloc[i, j]
                if abs(valor_corr) > 0.7:
                    correlacoes_fortes.append((matriz_corr.columns[i], matriz_corr.columns[j], valor_corr))

        if correlacoes_fortes:
            print("Correlações fortes (|r| > 0.7):")
            for var1, var2, corr in correlacoes_fortes:
                print(f"  {var1} vs {var2}: {corr:.3f}")
        else:
            print("Nenhuma correlação forte encontrada.")

    num_periodos = len(resultados_correlacao)
    if num_periodos > 0:
        if num_periodos <= 2:
            fig, axes = plt.subplots(1, num_periodos, figsize=(8 * num_periodos, 6))
        else:
            rows = (num_periodos + 1) // 2 
            cols = min(2, num_periodos)
            fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))

        if num_periodos > 1:
            axes = axes.ravel()
        else:
            axes = np.array([axes])

        for idx, (nome_intervalo, matriz_corr) in enumerate(resultados_correlacao.items()):
            if idx < len(axes):
                sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', center=0,
                            ax=axes[idx], fmt='.2f', cbar_kws={"shrink": .8})
                axes[idx].set_title(f'Correlações - {nome_intervalo}')

        plt.tight_layout()
        plt.savefig(os.path.join('Analise exploratoria', 'correlacoes_por_periodo.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        for nome_intervalo, matriz_corr in resultados_correlacao.items():
            matriz_corr.to_csv(os.path.join('Analise exploratoria',
                                            f'correlacao_{nome_intervalo.replace("-", "_")}.csv'))

    return resultados_correlacao

# Gera gráficos de dispersão para pares de variáveis mais correlacionadas
def graficos_dispersao(dados_df, max_combinacoes=6):
    print("\n" + "=" * 50)
    print("GRÁFICOS DE DISPERSÃO")
    print("=" * 50)
    colunas_numericas = dados_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Ano' in colunas_numericas:
        colunas_numericas.remove('Ano')

    if len(colunas_numericas) < 2:
        print("Número insuficiente de variáveis numéricas para gráficos de dispersão.")
        return

    matriz_corr = dados_df[colunas_numericas].corr()

    pares_correlacao = []
    for i in range(len(matriz_corr.columns)):
        for j in range(i + 1, len(matriz_corr.columns)):
            valor_corr = matriz_corr.iloc[i, j]
            if abs(valor_corr) < 0.95:  # Filtrar variáveis com correlação inferior a 0.95
                pares_correlacao.append(
                    (matriz_corr.columns[i], matriz_corr.columns[j], abs(valor_corr), valor_corr))

    if not pares_correlacao:
        print("Nenhum par de variáveis com correlação inferior a 0.95 foi encontrado.")
        return

    pares_correlacao.sort(key=lambda x: x[2], reverse=True)

    top_pares = pares_correlacao[:max_combinacoes]

    print(f"Gerando gráficos de dispersão para as {len(top_pares)} combinações com maior correlação:")

    num_pares = len(top_pares)
    num_cols = 3
    num_rows = (num_pares + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 5 * num_rows))

    for idx, (var1, var2, abs_corr, corr) in enumerate(top_pares, 1):
        plt.subplot(num_rows, num_cols, idx)
        plt.scatter(dados_df[var1], dados_df[var2], alpha=0.6, s=30)

        if not dados_df[var1].isna().all() and not dados_df[var2].isna().all():
            dados_limpos = dados_df[[var1, var2]].dropna()
            if len(dados_limpos) > 1:
                z = np.polyfit(dados_limpos[var1], dados_limpos[var2], 1)
                p = np.poly1d(z)
                x_range = np.linspace(dados_limpos[var1].min(), dados_limpos[var1].max(), 100)
                plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)

        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.title(f'{var1} vs {var2}\nCorrelação: {corr:.3f}')
        plt.grid(True, alpha=0.3)
    
        print(f"  {var1} vs {var2}: r = {corr:.3f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join('Analise exploratoria', 'graficos_dispersao.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


#Função principal
def main():
    print("INICIANDO ANÁLISE EXPLORATÓRIA DE DADOS")
    print("=" * 50)

    dados_df = carregar_dados()
    if dados_df is None:
        return
    info_basicas = informacoes_basicas(dados_df)
    df_valores_em_falta = analisar_valores_em_falta(dados_df)
    estatisticas_desc, colunas_numericas = estatisticas_descritivas(dados_df)
    caracteristicas_redundantes = identificar_caracteristicas_redundantes(dados_df)
    crescimento_regiao = analise_crescimento_por_regiao(dados_df)
    resultados_crescimento = municipios_maior_crescimento(dados_df)
    if resultados_crescimento:
        evolucao_temporal_maior_crescimento(dados_df, resultados_crescimento)

    print("\n" + "=" * 50)
    print("CONFIGURAÇÃO DA ANÁLISE DE CORRELAÇÃO POR INTERVALOS DE TEMPO")
    print("=" * 50)

    try:
        num_intervalos = int(input("Digite o número de intervalos desejado (ex: 4): "))
    except ValueError:
        print("Valor inválido! Usando o valor padrão: 4")
        num_intervalos = 4

    print("Digite os intervalos no formato 'Ano1-Ano2' (um por linha)")
    print("Exemplo: 2009-2012, 2013-2016, 2017-2020, 2021-2023")
    print("Pressione Enter com linha vazia para finalizar")

    intervalos_personalizados = []
    while True:
        intervalo = input(f"Intervalo {len(intervalos_personalizados) + 1}: ")
        if not intervalo:
            break
        intervalos_personalizados.append(intervalo)

    if not intervalos_personalizados:
        anos_disponiveis = sorted(dados_df['Ano'].unique())
        if len(anos_disponiveis) >= 2:
            ano_min = min(anos_disponiveis)
            ano_max = max(anos_disponiveis)
            intervalo_anos = ano_max - ano_min
            tamanho_intervalo = intervalo_anos // num_intervalos

            intervalos_personalizados = []
            for i in range(num_intervalos):
                inicio = ano_min + i * tamanho_intervalo
                fim = ano_min + (i + 1) * tamanho_intervalo - 1
                if i == num_intervalos - 1:
                    fim = ano_max
                intervalos_personalizados.append(f"{inicio}-{fim}")

            print("Intervalos gerados automaticamente:")
            for intervalo in intervalos_personalizados:
                print(f"  {intervalo}")
        else:
            print("Dados insuficientes para gerar intervalos. Usando valores padrão.")
            intervalos_personalizados = ["2009-2012, 2013-2016, 2017-2020, 2021-2023"]

    correlacoes_periodos = correlacao_por_intervalos_tempo(dados_df, num_intervalos, intervalos_personalizados)
    graficos_dispersao(dados_df)

    if colunas_numericas:
        informacao_outliers = detectar_outliers(dados_df, colunas_numericas)
        matriz_corr, correlacoes_fortes = matriz_correlacao(dados_df, colunas_numericas)
        distribuicoes_variaveis(dados_df, colunas_numericas)
        dados_temporais = analise_temporal(dados_df)


    print("\nANÁLISE EXPLORATÓRIA CONCLUÍDA!")
    print(f"Resultados salvos na pasta: {caminho_saida_base}")


if __name__ == "__main__":
    main()