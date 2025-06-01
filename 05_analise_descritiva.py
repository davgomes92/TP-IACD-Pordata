import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import warnings

warnings.filterwarnings('ignore')

sns.set_palette("husl")
plt.style.use('seaborn-v0_8')

#Dicionario que mapeia os municipios por região
Regioes_Municipios = {
    "Entre Douro e Minho": [
        "Amares", "Arcos de Valdevez", "Barcelos", "Braga", "Caminha", "Celorico de Basto",
        "Esposende", "Fafe", "Guimarães", "Melgaço", "Monção", "Mondim de Basto",
        "Paredes de Coura", "Ponte da Barca", "Ponte de Lima", "Póvoa de Lanhoso",
        "Ribeira de Pena", "Terras de Bouro", "Valença", "Viana do Castelo",
        "Vila Nova de Cerveira", "Vila Verde", "Vizela", "Amarante", "Baião", "Felgueiras",
        "Gondomar", "Lousada", "Maia", "Marco de Canaveses", "Matosinhos", "Paços de Ferreira",
        "Paredes", "Penafiel", "Porto", "Póvoa de Varzim", "Santo Tirso", "Trofa",
        "Valongo", "Vila do Conde", "Vila Nova de Gaia", "Cinfães", "Vieira do Minho", "Vila Nova de Famalicão"
    ],
    "Trás-os-Montes e Alto Douro": [
        "Alfândega da Fé", "Alijó", "Boticas", "Bragança", "Carrazeda de Ansiães",
        "Chaves", "Freixo de Espada à Cinta", "Macedo de Cavaleiros", "Mesão Frio",
        "Miranda do Douro", "Mirandela", "Mogadouro", "Montalegre", "Murça",
        "Peso da Régua", "Resende", "Sabrosa", "Santa Marta de Penaguião",
        "Valpaços", "Vila Flor", "Vila Pouca de Aguiar", "Vila Real", "Vimioso",
        "Vinhais", "Armamar", "Lamego", "Moimenta da Beira", "Penedono",
        "São João da Pesqueira", "Sernancelhe", "Tabuaço", "Arouca", "Vila Nova de Foz Côa", "Cabeceiras de Basto",
        "Tarouca", "Torre de Moncorvo"
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
        "Vila Velha de Ródão", "Castelo de Paiva", "Oliveira do Hospital", "Vila Nova de Paiva", "Sátão"
    ],
    "Estremadura e Ribatejo": [
        "Alcobaça", "Abrantes", "Alcanena", "Alenquer", "Almeirim", "Alpiarça", "Arruda dos Vinhos",
        "Azambuja", "Benavente", "Bombarral", "Cadaval", "Caldas da Rainha", "Cartaxo",
        "Chamusca", "Constância", "Coruche", "Entroncamento", "Ferreira do Zêzere",
        "Golegã", "Lourinhã", "Nazaré", "Óbidos", "Ourém", "Peniche",
        "Rio Maior", "Salvaterra de Magos", "Santarém", "Sobral de Monte Agraço",
        "Tomar", "Torres Novas", "Torres Vedras", "Vila Nova da Barquinha", "Mação", "Sardoal",
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
        "Velas", "Vila do Porto", "Vila Franca do Campo", "Vila da Praia da Vitória"
    ]
}

caminho_saida_base = "Analise Descritiva"
caminho_entrada_base = "Dados limpos"
nome_ficheiro_entrada ="dados_limpos_preprocessados.csv"
caminho_completo_entrada = os.path.join(caminho_entrada_base, nome_ficheiro_entrada)

os.makedirs(caminho_saida_base, exist_ok=True)

#Função para facilitar o processo de guardar as figuras
def guardar_fig(nome):
    plt.tight_layout()
    plt.savefig(os.path.join(caminho_saida_base, nome), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

#Função para mapear os municipios atraves do dicionario
def mapear_municipio_para_regiao(municipio):
    for regiao, municipios in Regioes_Municipios.items():
        if municipio in municipios:
            return regiao
    return "Região Desconhecida"

#Função para adicionar a coluna "Regiao" ao Datatframe
def adicionar_coluna_regiao(df):
    if 'Regiao' not in df.columns and 'Municipio' in df.columns:
        df['Regiao'] = df['Municipio'].apply(mapear_municipio_para_regiao)
    return df

#Função que carrega o CSV
def carregar_dados(caminho):
    try:
        df = pd.read_csv(caminho, encoding='utf-8')
        return adicionar_coluna_regiao(df)
    except Exception as erro:
        print(f"Erro ao carregar dados: {erro}")
        return None

#Função que deteta se existe uma coluna referente a Regiao
def detetar_coluna_regiao(df):
    for col in ['Distrito', 'Regiao', 'NUTS', 'Zona']:
        if col in df.columns:
            return col
    return None

#Extrai as colunas numericas que não sejam ano
def extrair_variaveis_numericas(df):
    colunas = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in colunas if col not in ['Ano', 'ano', 'Year']]

#Calculo da Volatilidade
def calcular_volatilidade(series):
    mean_value = np.nanmean(series)
    std_dev = np.nanstd(series)
    if mean_value == 0:
        return np.nan
    return (std_dev / mean_value) * 100

#Normalização dos dados
def normalizar_dados(df, colunas):
    scaler = StandardScaler()
    return scaler.fit_transform(df[colunas]), scaler

#Calculo da reta mais adequada para o conjunto
def calcular_tendencia_linear(anos, valores):
    modelo = LinearRegression()
    modelo.fit(anos, valores)
    return modelo.predict(anos), modelo.coef_[0], modelo.score(anos, valores)

#Obter o numero ideial de cluster Silhoutte score (Mede quão juntos estão os pontos de dados dentro de um cluster)
def obter_numero_clusters_otimo(X, max_k=10):
    scores = []
    for k in range(2, min(max_k + 1, len(X))):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(X)
        score = silhouette_score(X, clusters)
        scores.append((k, score))
    return max(scores, key=lambda x: x[1])[0]

#Analise estatistica basica
def analise_estatistica_basica(df):
    variaveis = extrair_variaveis_numericas(df)
    print(f"\nVariáveis numéricas encontradas: {variaveis}")
    resumo = df[variaveis].describe().round(2)
    print(resumo)
    assimetria = df[variaveis].skew().round(2)
    curtose = df[variaveis].kurtosis().round(2)
    print("\nAssimetria:")
    print(assimetria)
    print("\nCurtose:")
    print(curtose)
    return resumo

#Função que faz uma análise temporal de variáveis numéricas
def analise_temporal(df):
    variaveis = extrair_variaveis_numericas(df)
    for var in variaveis:
        dados_ano = df.groupby('Ano')[var].agg(['mean', 'std']).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(dados_ano['Ano'], dados_ano['mean'], 'o-', label='Média')
        ax1.fill_between(dados_ano['Ano'],
                         dados_ano['mean'] - dados_ano['std'],
                         dados_ano['mean'] + dados_ano['std'],
                         alpha=0.3, label='±1 Desvio Padrão')

        pred, slope, r2 = calcular_tendencia_linear(
            dados_ano['Ano'].values.reshape(-1, 1), dados_ano['mean'].values)
        ax1.plot(dados_ano['Ano'], pred, '--', label=f'Tendência (R²={r2:.2f})')
        ax1.set_title(f'Evolução Temporal - {var}')
        ax1.set_xlabel('Ano')
        ax1.set_ylabel(var)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        volatilidade = dados_ano.set_index('Ano')['std'] / dados_ano.set_index('Ano')['mean']
        ax2.plot(dados_ano['Ano'], volatilidade, 'o-', color='red')
        ax2.set_title(f'Volatilidade - {var}')
        ax2.set_xlabel('Ano')
        ax2.set_ylabel('Coeficiente de Variação')
        ax2.grid(True, alpha=0.3)

        guardar_fig(f"temporal_completo_{var}.png")

#Mostra o crescimento da variavel por Região
def crescimento_por_regiao(df):
    col_regiao = detectar_coluna_regiao(df)
    if not col_regiao:
        print("Coluna de região não encontrada.")
        return

    variaveis = extrair_variaveis_numericas(df)
    anos = sorted(df['Ano'].unique())
    ini, fim = anos[0], anos[-1]

    dados_ini = df[df['Ano'] == ini].groupby(col_regiao)[variaveis].mean()
    dados_fim = df[df['Ano'] == fim].groupby(col_regiao)[variaveis].mean()

    crescimento = ((dados_fim - dados_ini) / dados_ini * 100).fillna(0).round(2)

    for var in variaveis:
        plt.figure(figsize=(12, 6))
        crescimento[var].plot(kind='bar', color=['green' if x > 0 else 'red' for x in crescimento[var]])
        plt.title(f'Crescimento por {col_regiao} - {var}')
        plt.ylabel('% Crescimento')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        guardar_fig(f"crescimento_regiao_{var}.png")

    return crescimento

#Mostra os municipios com maior crescimento e maior decrescimo
def municipios_maior_crescimento(df, top_n=10):
    variaveis = extrair_variaveis_numericas(df)
    anos = sorted(df['Ano'].unique())
    ini, fim = anos[0], anos[-1]

    dados_ini = df[df['Ano'] == ini].set_index('Municipio')[variaveis]
    dados_fim = df[df['Ano'] == fim].set_index('Municipio')[variaveis]

    crescimento = ((dados_fim - dados_ini) / dados_ini * 100).fillna(0).round(2)

    for var in variaveis:
        maiores = crescimento[var].nlargest(top_n)
        menores = crescimento[var].nsmallest(top_n)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        maiores.plot(kind='barh', color='green', ax=ax1)
        ax1.set_title(f'Top {top_n} Crescimento - {var}')
        ax1.set_xlabel('% Crescimento')

        menores.plot(kind='barh', color='red', ax=ax2)
        ax2.set_title(f'Top {top_n} Decrescimento - {var}')
        ax2.set_xlabel('% Crescimento')

        guardar_fig(f"municipios_crescimento_{var}.png")

    return crescimento

#Analisa a correlação geral
def analise_correlacoes(df):
    variaveis = extrair_variaveis_numericas(df)

    corr = df[variaveis].corr()
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlações')
    guardar_fig("correlacoes_geral.png")


#Analisa a correlação por intervalo fornecido peloo utilizador
def matriz_correlacao_por_periodo(df):
    anos_disponiveis = sorted(df["Ano"].unique())
    ano_min = min(anos_disponiveis)
    ano_max = max(anos_disponiveis)
    
    print(f"\nAnos disponíveis no conjunto de dados: {ano_min} a {ano_max}")
    escolha = input("Deseja analisar todos os dados (T) ou definir intervalos específicos (I)? ").strip().upper()
    intervalos = []

    if escolha == 'T':
        intervalos = [(ano_min, ano_max)]
        print(f"A utilizar todo o período: {ano_min} a {ano_max}")

    elif escolha == 'I':
        continuar = True
        while continuar:
            try:
                print("\nDefina um intervalo de tempo (ambos os anos incluidos):")
                ano_inicio = int(input(f"Ano de início (entre {ano_min} e {ano_max}): "))
                ano_fim = int(input(f"Ano de fim (entre {ano_inicio} e {ano_max}): "))

                if ano_inicio < ano_min or ano_inicio > ano_max:
                    print(f"Erro: O ano de início deve estar entre {ano_min} e {ano_max}.")
                    continue

                if ano_fim < ano_inicio or ano_fim > ano_max:
                    print(f"Erro: O ano de fim deve estar entre {ano_inicio} e {ano_max}.")
                    continue

                intervalos.append((ano_inicio, ano_fim))

                mais_intervalos = input("Deseja adicionar outro intervalo? (S/N): ").strip().upper()
                continuar = (mais_intervalos == 'S')

            except ValueError:
                print("Erro: Por favor, insira anos válidos (números inteiros).")

    else:
        print("Opção inválida. A utilizar todo o período como padrão.")
        intervalos = [(ano_min, ano_max)]

    variaveis_disponiveis = extrair_variaveis_numericas(df)

    print("\nVariáveis disponíveis para análise:")
    for idx, var in enumerate(variaveis_disponiveis, 1):
        print(f"{idx}. {var}")

    escolha_vars = input(
        "Quais variáveis deseja incluir? (Digite os números separados por vírgula, ou 'T' para todas): ").strip()

    if escolha_vars.upper() == 'T':
        vars_selecionadas = variaveis_disponiveis
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in escolha_vars.split(',')]
            vars_selecionadas = [variaveis_disponiveis[i] for i in indices if 0 <= i < len(variaveis_disponiveis)]
        except:
            print("Seleção inválida. A utilizar todas as variáveis.")
            vars_selecionadas = variaveis_disponiveis

    resultado = {}

    for inicio, fim in intervalos:
        dados_periodo = df[(df["Ano"] >= inicio) &
                              (df["Ano"] <= fim)]

        dados_filtrados = df[vars_selecionadas]

        matriz_corr = dados_filtrados.corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
        sns.heatmap(matriz_corr, mask=mask, annot=True, cmap='coolwarm',
                    fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)

        titulo = f'Matriz de Correlação ({inicio}-{fim})'
        plt.title(titulo, fontsize=14)

        nome_arquivo = f'matriz_correlacao_{inicio}_{fim}.png'
        guardar_fig(nome_arquivo)

        resultado[f'{inicio}_{fim}'] = {
            'matriz': matriz_corr,
            'arquivo': nome_arquivo
        }

        print(f"\nAnálise de correlação para o período {inicio}-{fim} concluída.")
        print(f"Matriz guardada como {nome_arquivo}")

        salvar_csv = input("Deseja guardar a matriz de correlação em CSV? (S/N): ").strip().upper()
        if salvar_csv == 'S':
            nome_csv = f'matriz_correlacao_{inicio}_{fim}.csv'
            matriz_corr.to_csv(caminho_saida_base + "/" + nome_csv)
            print(f"Matriz guardada em formato CSV como {nome_csv}")
    return resultado

#Analise e criação de Cluster através do metódo K-Means
def analise_clustering(df, max_clusters=8):
    variaveis = extrair_variaveis_numericas(df)
    ano_max = df['Ano'].max()
    dados = df[df['Ano'] == ano_max].dropna(subset=variaveis)
    if len(dados) < 4:
        print("Dados insuficientes para clustering.")
        return
    X, scaler = normalizar_dados(dados, variaveis)

    n_clusters = obter_numero_clusters_otimo(X, max_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    dados['Cluster'] = kmeans.fit_predict(X)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    sns.countplot(x='Cluster', data=dados, ax=axes[0, 0])
    axes[0, 0].set_title('Distribuição por Cluster')

    media = dados.groupby('Cluster')[variaveis].mean()
    media.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Médias por Cluster')
    axes[0, 1].tick_params(axis='x', rotation=45)

    if len(variaveis) >= 3:
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        for cluster in dados['Cluster'].unique():
            cluster_data = dados[dados['Cluster'] == cluster]
            ax.scatter(cluster_data[variaveis[0]], cluster_data[variaveis[1]], cluster_data[variaveis[2]],
                       label=f'Cluster {cluster}', alpha=0.7)
        ax.set_xlabel(variaveis[0])
        ax.set_ylabel(variaveis[1])
        ax.set_zlabel(variaveis[2])
        ax.set_title('Clusters - 3D Scatter Plot')
        ax.legend()

    scores = []
    k_range = range(2, min(max_clusters + 1, len(dados)))
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans_temp.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)

    axes[1, 1].plot(k_range, scores, 'o-')
    axes[1, 1].axvline(x=n_clusters, color='red', linestyle='--', label=f'Ótimo: {n_clusters}')
    axes[1, 1].set_xlabel('Número de Clusters')
    axes[1, 1].set_ylabel('Silhouette Score')
    axes[1, 1].set_title('Análise do Número de Clusters')
    axes[1, 1].legend()

    guardar_fig("clustering_completo.png")

    return dados

#Cria um ranking para os municipios de destaque
def analise_rankings(df):
    variaveis = extrair_variaveis_numericas(df)
    ano_recente = df['Ano'].max()
    dados_recentes = df[df['Ano'] == ano_recente]

    for var in variaveis:
        top_10 = dados_recentes.nlargest(10, var)[['Municipio', var]]
        bottom_10 = dados_recentes.nsmallest(10, var)[['Municipio', var]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        top_10.plot(x='Municipio', y=var, kind='bar', ax=ax1, color='green')
        ax1.set_title(f'Top 10 - {var} ({ano_recente})')
        ax1.tick_params(axis='x', rotation=45)

        bottom_10.plot(x='Municipio', y=var, kind='bar', ax=ax2, color='red')
        ax2.set_title(f'Bottom 10 - {var} ({ano_recente})')
        ax2.tick_params(axis='x', rotation=45)

        guardar_fig(f"ranking_{var}.png")

#Mostra a evolução temporal por Região
def mostrar_evolucao_temporal_por_regiao(df):
    col_regiao = "Regiao"
    col_ano = "Ano"
    if col_ano not in df.columns or col_regiao not in df.columns:
        raise ValueError(f"As colunas '{col_ano}' e '{col_regiao}' devem estar presentes no DataFrame.")

    colunas_numericas = df.select_dtypes(include=['number']).columns
    colunas_numericas = [col for col in colunas_numericas if col != col_ano]

    if not colunas_numericas:
        raise ValueError("Não há colunas numéricas, além do 'Ano', para plotar.")

    for coluna in colunas_numericas:
        plt.figure(figsize=(10, 6))


        sns.lineplot(data=df, x=col_ano, y=coluna, hue=col_regiao, marker="o")
        plt.title(f"Evolução temporal de '{coluna}' por {col_regiao}", fontsize=14)
        plt.xlabel(col_ano, fontsize=12)
        plt.ylabel(coluna, fontsize=12)
        plt.legend(title=col_regiao, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        guardar_fig(f"Evolução temporal_por_{coluna}.png")

#Compara as Regiões
def comparacao_regioes_detalhada(df):
    col_regiao = detectar_coluna_regiao(df)
    if not col_regiao:
        return

    variaveis = extrair_variaveis_numericas(df)

    for var in variaveis:
        plt.figure(figsize=(14, 8))
        df.boxplot(column=var, by=col_regiao, figsize=(14, 8))
        plt.suptitle('')
        plt.title(f'Distribuição de {var} por {col_regiao}')
        plt.xticks(rotation=45)
        guardar_fig(f"distribuicao_regioes_{var}.png")

#Evolução temporal dos municipios de maior crescimento
def evolucao_temporal_maior_crescimento(df, top_n=10):
    variaveis = extrair_variaveis_numericas(df)
    anos = sorted(df['Ano'].unique())
    ini, fim = anos[0], anos[-1]

    dados_ini = df[df['Ano'] == ini].set_index('Municipio')[variaveis]
    dados_fim = df[df['Ano'] == fim].set_index('Municipio')[variaveis]
    crescimento = ((dados_fim - dados_ini) / dados_ini * 100).fillna(0).round(2)

    cores_regioes = {
        'Entre Douro e Minho': '#1f77b4',
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

    print(f"\nEvolução Temporal - Municípios com Maior Crescimento ({ini}-{fim})")
    print("=" * 60)

    for var in variaveis:
        top_municipios = crescimento[var].nlargest(top_n)
        municipios_selecionados = top_municipios.index.tolist()

        municipios_info = []
        for municipio in municipios_selecionados:
            regiao = mapear_municipio_para_regiao(municipio)
            crescimento_pct = top_municipios[municipio]
            municipios_info.append({'Municipio': municipio, 'Regiao': regiao, 'Crescimento': crescimento_pct})

        municipios_df = pd.DataFrame(municipios_info)

        print(f"\nTop {top_n} - {var}:")
        for _, row in municipios_df.iterrows():
            print(f"  {row['Municipio']} ({row['Regiao']}): {row['Crescimento']:.1f}%")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        for municipio in municipios_selecionados:
            dados_municipio = df[df['Municipio'] == municipio].sort_values('Ano')
            if not dados_municipio.empty:
                regiao = mapear_municipio_para_regiao(municipio)
                cor = cores_regioes.get(regiao, '#000000')

                ax1.plot(dados_municipio['Ano'], dados_municipio[var],
                         marker='o', label=f'{municipio} ({regiao})',
                         linewidth=2, color=cor, markersize=6)

        ax1.set_title(f'Evolução Temporal dos Top {top_n} Municípios - {var}', fontsize=14)
        ax1.set_xlabel('Ano')
        ax1.set_ylabel(var)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        regioes_count = municipios_df['Regiao'].value_counts()
        cores_bars = [cores_regioes.get(reg, '#000000') for reg in regioes_count.index]

        ax2.bar(range(len(regioes_count)), regioes_count.values, color=cores_bars)
        ax2.set_title(f'Distribuição Regional dos Top {top_n} Municípios - {var}')
        ax2.set_xlabel('Região')
        ax2.set_ylabel('Número de Municípios')
        ax2.set_xticks(range(len(regioes_count)))
        ax2.set_xticklabels(regioes_count.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        guardar_fig(f"evolucao_temporal_top_municipios_{var}.png")

#Prever a evolução das metricas, por região
def prever_evolucao_x_anos(df):
    try:
        anos_futuros_input = int(input("Indique a quantidade de anos que quer prever: "))
        if anos_futuros_input <= 0:
            raise ValueError("O número de anos deve ser maior que zero.")
    except ValueError as e:
        print(f"Erro: {e}")
        return

    ultimo_ano = int(df["Ano"].max())

    vars_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if "Ano" in vars_numericas:
        vars_numericas.remove("Ano")

    anos_futuros = list(range(ultimo_ano + 1, ultimo_ano + 1 + anos_futuros_input))
    previsoes = pd.DataFrame({"Ano": anos_futuros})

    entidade_col = None
    for col in ['Regiao']:
        if col in df.columns:
            entidade_col = col
            break
        for entidade in df[entidade_col].unique():
            df_entidade = df[df[entidade_col] == entidade].copy()

            for var in vars_numericas:
                if var not in df.columns:
                    continue

                X = df_entidade['Ano'].values.reshape(-1, 1)
                y = df_entidade[var].values

                mask = ~np.isnan(y)
                X = X[mask]
                y = y[mask]

                if len(y) < 2:
                    continue

                modelo = LinearRegression()
                modelo.fit(X, y)

                X_futuro = np.array(anos_futuros).reshape(-1, 1)
                y_previsto = modelo.predict(X_futuro)

                coluna_nome = f"{entidade}_{var}"
                previsoes[coluna_nome] = y_previsto

        entidades = df[entidade_col].unique()

        for entidade in entidades:
            plt.figure(figsize=(12, 8))
            df_entidade = df[df[entidade_col] == entidade]

            for var in vars_numericas:
                if var in df.columns:
                    plt.plot(df_entidade['Ano'], df_entidade[var],
                             marker='o', linewidth=2, markersize=4,
                             label=f'{var} (histórico)')

                    col_previsao = f"{entidade}_{var}"
                    if col_previsao in previsoes.columns:
                        plt.plot(previsoes['Ano'], previsoes[col_previsao],
                                 marker='x', linestyle='--', linewidth=2, markersize=6,
                                 label=f'{var} (previsão)')

            plt.title(f'Previsão de Evolução para {entidade} - Próximos {anos_futuros_input} Anos',
                      fontsize=14, fontweight='bold')
            plt.xlabel("Ano", fontsize=12)
            plt.ylabel('Valor', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            try:
                guardar_fig(f"previsao_{entidade}_{anos_futuros_input}_anos.png")
            except NameError:
                plt.savefig(f"previsao_{entidade}_{anos_futuros_input}_anos.png", dpi=300, bbox_inches='tight')

            plt.show()


    print(f"Previsão para os próximos {anos_futuros_input} anos concluída através do método Linear.")
    print(f"Variáveis analisadas: {vars_numericas}")
    if entidade_col:
        print(f"Gráficos criados para cada {entidade_col}: {list(df[entidade_col].unique())}")
        print(f"Total de gráficos gerados: {len(df[entidade_col].unique())}")

    return previsoes


def estatisticas_variadas(df):
    variaveis = extrair_variaveis_numericas(df)
    anos = sorted(df['Ano'].unique())
    ini, fim = anos[0], anos[-1]

    print(f"\nEstatísticas variadas ({ini}-{fim}):")
    print("=" * 50)
    for var in variaveis:
        serie_completa = df.groupby('Ano')[var].mean()
        _, slope, r2 = calcular_tendencia_linear(
            np.array(anos).reshape(-1, 1), serie_completa.values)
        volatilidade = calcular_volatilidade(serie_completa)
        crescimento_total = ((serie_completa.iloc[-1] - serie_completa.iloc[0]) /
                             serie_completa.iloc[0] * 100)
        print(f"\n{var}:")
        print(f"  Crescimento total: {crescimento_total:.1f}%")
        print(f"  Tendência anual: {slope:.2f} ({r2:.2f} R²)")
        print(f"  Volatilidade: {volatilidade:.2f}")
        if volatilidade > 0.3:
            print(f"  Status: Alta volatilidade")
        elif abs(slope) > serie_completa.mean() * 0.01:
            print(f"  Status: Tendência forte")
        else:
            print(f"  Status: Estável")

#Executa todas as analises
def main():
    df = carregar_dados(caminho_completo_entrada)
    if df is None or df.empty:
        print("Dados não carregados.")
        return

    print("\n[1] Estatística Básica")
    analise_estatistica_basica(df)

    print("\n[2] Análise Temporal Avançada")
    analise_temporal(df)

    print("\n[3] Crescimento por Região")
    crescimento_por_regiao(df)

    print("\n[4] Evolução por Região")
    mostrar_evolucao_temporal_por_regiao(df)

    print("\n[5] Rankings Municipais")
    analise_rankings(df)

    print("\n[6] Crescimento Detalhado")
    municipios_maior_crescimento(df)

    print("\n[7] Evolução dos Municipios com maior crescimento")
    evolucao_temporal_maior_crescimento(df, top_n=10)

    print("\n[8] Correlações Avançadas")
    analise_correlacoes(df)

    print("\n[9] Correlações Por Período")
    matriz_correlacao_por_periodo(df)

    print("\n[10] Comparação Regional")
    comparacao_regioes_detalhada(df)

    print("\n[11] Previsão")
    prever_evolucao_x_anos(df)

    print("\n[12] Clustering Otimizado")
    print(analise_clustering(df))

    print("\n[13] Estatistícas Variadas")
    estatisticas_variadas(df)

# Execução
if __name__ == "__main__":
    main()

