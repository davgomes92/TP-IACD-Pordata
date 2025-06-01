import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

caminho_entrada_base = "Dados integrados"
caminho_saida_base = "Dados limpos"
caminho_analise = "Analise exploratoria"
nome_ficheiro_entrada = "dados_integrados_pordata.csv"
nome_ficheiro_saida = "dados_limpos_preprocessados.csv"

caminho_completo_entrada = os.path.join(caminho_entrada_base, nome_ficheiro_entrada)
caminho_completo_saida = os.path.join(caminho_saida_base, nome_ficheiro_saida)

os.makedirs(caminho_saida_base, exist_ok=True)

#Carrega o conjunto de Dados
def carregar_dados():
    try:
        print(f"A carregar dados de: {caminho_completo_entrada}")
        df = pd.read_csv(caminho_completo_entrada, encoding='utf-8')
        print(f"Dados carregados com sucesso. Dimensões: {df.shape}")
        return df
    except Exception as erro:
        print(f"ERRO ao carregar dados: {erro}")
        return None

#Identifica colunas numericas redundantes, exceto ano (correlação acima de 0.95)
def identificar_colunas_redundantes(df, limiar=0.95):
    print("\n" + "="*50)
    print("IDENTIFICAÇÃO DE COLUNAS REDUNDANTES")
    print("="*50)

    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Ano' in colunas_numericas:
        colunas_numericas.remove('Ano')
    
    if len(colunas_numericas) < 2:
        print("Número insuficiente de variáveis numéricas para análise de redundância.")
        return []

    matriz_correlacao = df[colunas_numericas].corr().abs()

    colunas_redundantes = []
    for i in range(len(matriz_correlacao.columns)):
        for j in range(i+1, len(matriz_correlacao.columns)):
            if matriz_correlacao.iloc[i, j] > limiar:
                col1, col2 = matriz_correlacao.columns[i], matriz_correlacao.columns[j]
                contagem_falta1 = df[col1].isna().sum()
                contagem_falta2 = df[col2].isna().sum()
                
                coluna_a_remover = col2 if contagem_falta1 <= contagem_falta2 else col1
                coluna_a_manter = col1 if contagem_falta1 <= contagem_falta2 else col2
                
                if coluna_a_remover not in colunas_redundantes:
                    colunas_redundantes.append(coluna_a_remover)
                    print(f"Redundância detectada: {col1} vs {col2} (r={matriz_correlacao.iloc[i, j]:.3f})")
                    print(f"  Será removida: {coluna_a_remover} (mantida: {coluna_a_manter})")
    
    if not colunas_redundantes:
        print("Nenhuma coluna redundante encontrada.")
    
    return colunas_redundantes

#Tratamento de Outliers atraves de Winsorização (para os de IQR)
def tratar_valores_extremos(df, fator=1.5):
    print("\n" + "="*50)
    print("TRATAMENTO DE VALORES EXTREMOS")
    print("="*50)
    dados_tratados = df.copy()
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Ano' in colunas_numericas:
        colunas_numericas.remove('Ano')
    info_extremos = {}
    
    for coluna in colunas_numericas:
        contagem_original = dados_tratados[coluna].count()

        q1 = dados_tratados[coluna].quantile(0.25)
        q3 = dados_tratados[coluna].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - fator * iqr
        limite_superior = q3 + fator * iqr

        mascara_extremos = (dados_tratados[coluna] < limite_inferior) | (dados_tratados[coluna] > limite_superior)
        contagem_extremos = mascara_extremos.sum()
            
        dados_tratados.loc[dados_tratados[coluna] < limite_inferior, coluna] = limite_inferior
        dados_tratados.loc[dados_tratados[coluna] > limite_superior, coluna] = limite_superior
        
        info_extremos[coluna] = {
            'contagem_original': contagem_original,
            'extremos_tratados': contagem_extremos,
            'percentagem': (contagem_extremos / contagem_original) * 100 if contagem_original > 0 else 0
        }
        if contagem_extremos > 0:
            print(f"{coluna}: {contagem_extremos} valores extremos tratados ({info_extremos[coluna]['percentagem']:.2f}%)")
    
    return dados_tratados, info_extremos

#Trata os valores em falta
def tratar_valores_em_falta(df, estrategia='auto'):
    print("\n" + "="*50)
    print("TRATAMENTO DE VALORES EM FALTA")
    print("="*50)
    
    dados_tratados = df.copy()
    info_em_falta = {}
    contagens_em_falta = dados_tratados.isnull().sum()
    colunas_em_falta = contagens_em_falta[contagens_em_falta > 0].index.tolist()
    if not colunas_em_falta:
        print("Nenhum valor em falta encontrado.")
        return dados_tratados, info_em_falta
    
    for coluna in colunas_em_falta:
        contagem_faltante = contagens_em_falta[coluna]
        percentagem_faltante = (contagem_faltante / len(dados_tratados)) * 100
        
        print(f"\n{coluna}: {contagem_faltante} valores em falta ({percentagem_faltante:.2f}%)")
        if percentagem_faltante > 50:
            print(f"  ATENÇÃO: {coluna} tem mais de 50% de valores em falta. Considerar remoção.")
            estrategia_usada = "considerar_remocao"
            valor_preenchimento = None
        elif coluna in dados_tratados.select_dtypes(include=[np.number]).columns:
            if estrategia == 'auto':
                if percentagem_faltante < 5:
                    valor_preenchimento = dados_tratados[coluna].mean()
                    estrategia_usada = "media"
                else:
                    valor_preenchimento = dados_tratados[coluna].median()
                    estrategia_usada = "mediana"
            else:
                if estrategia == 'media':
                    valor_preenchimento = dados_tratados[coluna].mean()
                elif estrategia == 'mediana':
                    valor_preenchimento = dados_tratados[coluna].median()
                estrategia_usada = estrategia
            
            dados_tratados[coluna].fillna(valor_preenchimento, inplace=True)
        
        else:
            if percentagem_faltante < 10:
                valor_preenchimento = dados_tratados[coluna].mode().iloc[0] if not dados_tratados[coluna].mode().empty else "Desconhecido"
                dados_tratados[coluna].fillna(valor_preenchimento, inplace=True)
                estrategia_usada = "moda"
            else:
                valor_preenchimento = "Desconhecido"
                dados_tratados[coluna].fillna(valor_preenchimento, inplace=True)
                estrategia_usada = "categoria_desconhecida"
        
        info_em_falta[coluna] = {
            'em_falta_originais': contagem_faltante,
            'percentagem': percentagem_faltante,
            'estrategia': estrategia_usada,
            'valor_preenchimento': valor_preenchimento
        }
        
        print(f"  Estratégia aplicada: {estrategia_usada}")
    
    return dados_tratados, info_em_falta

#Remove as colunas desnecessarias
def remover_colunas_desnecessarias(df, colunas_redundantes, limiar_em_falta_alto=0.5):
    print("\n" + "="*50)
    print("REMOÇÃO DE COLUNAS DESNECESSÁRIAS")
    print("="*50)
    
    dados_limpos = df.copy()
    colunas_removidas = []
    for coluna in colunas_redundantes:
        if coluna in dados_limpos.columns:
            dados_limpos.drop(columns=[coluna], inplace=True)
            colunas_removidas.append(coluna)
            print(f"Coluna redundante removida: {coluna}")

    percentagens_em_falta = dados_limpos.isnull().sum() / len(dados_limpos)
    colunas_muitos_em_falta = percentagens_em_falta[percentagens_em_falta > limiar_em_falta_alto].index.tolist()
    
    for coluna in colunas_muitos_em_falta:
        if coluna not in ['Ano', 'Municipio']:  # Não remover colunas identificadoras
            dados_limpos.drop(columns=[coluna], inplace=True)
            colunas_removidas.append(coluna)
            print(f"Coluna com muitos valores em falta removida: {coluna} ({percentagens_em_falta[coluna]*100:.1f}%)")

    colunas_numericas = dados_limpos.select_dtypes(include=[np.number]).columns
    for coluna in colunas_numericas:
        if dados_limpos[coluna].var() == 0:
            dados_limpos.drop(columns=[coluna], inplace=True)
            colunas_removidas.append(coluna)
            print(f"Coluna com variância zero removida: {coluna}")
    
    if not colunas_removidas:
        print("Nenhuma coluna foi removida.")
    
    return dados_limpos, colunas_removidas

#Encontrar inconcistencias no DataFrame
def encontrar_inconcistencias(df):
    print("\n" + "=" * 50)
    print("DETECÇÃO DE INCONSISTÊNCIAS")
    print("=" * 50)

    inconsistencias = {}

    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Ano' in colunas_numericas:
        colunas_numericas.remove('Ano')

    for coluna in colunas_numericas:
        contagem_negativos = (df[coluna] < 0).sum()
        if contagem_negativos > 0:
            inconsistencias[f"{coluna}_negativos"] = {
                'tipo': 'valores_negativos',
                'contagem': contagem_negativos,
                'descricao': f'{coluna} tem {contagem_negativos} valores negativos'
            }
            print(f"INCONSISTÊNCIA: {coluna} tem {contagem_negativos} valores negativos")

    duplicadas = df.duplicated().sum()
    if duplicadas > 0:
        inconsistencias['duplicadas'] = {
            'tipo': 'linhas_duplicadas',
            'contagem': duplicadas,
            'descricao': f'Encontradas {duplicadas} linhas duplicadas'
        }
        print(f"INCONSISTÊNCIA: {duplicadas} linhas duplicadas encontradas")

    if 'Ano' in df.columns:
        anos_invalidos = df[(df['Ano'] < 1900) | (df['Ano'] > 2030)]['Ano'].count()
        if anos_invalidos > 0:
            inconsistencias['anos_invalidos'] = {
                'tipo': 'anos_invalidos',
                'contagem': anos_invalidos,
                'descricao': f'Encontrados {anos_invalidos} anos inválidos'
            }
            print(f"INCONSISTÊNCIA: {anos_invalidos} anos inválidos encontrados")

    if 'Municipio' in df.columns:
        municipios_vazios = df['Municipio'].isnull().sum()
        contagem_portugal_municipio = (df['Municipio'] == "Portugal").sum()

        if municipios_vazios > 0:
            inconsistencias['municipios_vazios'] = {
                'tipo': 'municipios_vazios',
                'contagem': municipios_vazios,
                'descricao': f'Encontrados {municipios_vazios} Municipios vazios'
            }
            print(f"INCONSISTÊNCIA: {municipios_vazios} Municipios vazios")

        if contagem_portugal_municipio > 0:
            inconsistencias['portugal_como_municipio'] = {
                'tipo': 'portugal_como_municipio',
                'contagem': contagem_portugal_municipio,
                'descricao': f'Encontrados {contagem_portugal_municipio} registros com "Portugal" como Municipio'
            }
            print(f"INCONSISTÊNCIA: {contagem_portugal_municipio} registros com 'Portugal' como Municipio")


    if not inconsistencias:
        print("Nenhuma inconsistência detectada.")

    return inconsistencias

#Corrige as inconcistencias encontradas
def corrigir_inconsistencias(df, inconsistencias):
    print("\n" + "="*50)
    print("CORREÇÃO DE INCONSISTÊNCIAS")
    print("="*50)
    dados_corrigidos = df.copy()
    correcoes_aplicadas = {}
    for chave, inconsistencia in inconsistencias.items():
        if inconsistencia['tipo'] == 'valores_negativos':
            coluna = chave.replace('_negativos', '')
            if coluna in dados_corrigidos.columns:
                mascara_negativos = dados_corrigidos[coluna] < 0
                dados_corrigidos.loc[mascara_negativos, coluna] = 0
                correcoes_aplicadas[chave] = f"Valores negativos em {coluna} substituídos por 0"
                print(f"Corrigido: Valores negativos em {coluna} substituídos por 0")

        elif inconsistencia['tipo'] == 'linhas_duplicadas':
            dados_corrigidos.drop_duplicates(inplace=True)
            correcoes_aplicadas[chave] = "Linhas duplicadas removidas"
            print("Corrigido: Linhas duplicadas removidas")

        elif inconsistencia['tipo'] == 'anos_invalidos':
            mascara_anos_validos = (dados_corrigidos['Ano'] >= 1900) & (dados_corrigidos['Ano'] <= 2030)
            dados_corrigidos = dados_corrigidos[mascara_anos_validos]
            correcoes_aplicadas[chave] = "Linhas com anos inválidos removidas"
            print("Corrigido: Linhas com anos inválidos removidas")

        elif inconsistencia['tipo'] == 'municipios_vazios':
            dados_corrigidos = dados_corrigidos[dados_corrigidos['Municipio'].notna()]
            correcoes_aplicadas[chave] = "Linhas com Municipios vazios removidas"
            print("Corrigido: Linhas com Municipios vazios removidas")

        elif inconsistencia['tipo'] == 'portugal_como_municipio':
            dados_corrigidos = dados_corrigidos[dados_corrigidos['Municipio'] != "Portugal"]
            correcoes_aplicadas[chave] = "Linhas com Municipios com nome Portugal removidas"
            print("Corrigido: Linhas com Municipios com nome Portugal removidas")

    if not correcoes_aplicadas:
        print("Nenhuma correção foi necessária.")
    
    return dados_corrigidos, correcoes_aplicadas

#Validação do processo
def validar_dados_finais(df):
    print("\n" + "="*50)
    print("VALIDAÇÃO DOS DADOS FINAIS")
    print("="*50)
    
    relatorio_validacao = {}

    print(f"Dimensões finais: {df.shape}")
    relatorio_validacao['dimensoes_finais'] = df.shape

    contagens_em_falta = df.isnull().sum()
    total_em_falta = contagens_em_falta.sum()
    print(f"Total de valores em falta: {total_em_falta}")
    relatorio_validacao['total_em_falta'] = total_em_falta
    
    if total_em_falta > 0:
        print("Valores em falta por coluna:")
        for coluna, contagem in contagens_em_falta[contagens_em_falta > 0].items():
            percentagem = (contagem / len(df)) * 100
            print(f"  {coluna}: {contagem} ({percentagem:.2f}%)")

    duplicadas = df.duplicated().sum()
    print(f"Linhas duplicadas: {duplicadas}")
    relatorio_validacao['duplicadas'] = duplicadas

    print("Tipos de dados:")
    for coluna, tipo in df.dtypes.items():
        print(f"  {coluna}: {tipo}")
    relatorio_validacao['tipos_dados'] = dict(df.dtypes)

    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if colunas_numericas:
        print("\nEstatísticas das variáveis numéricas:")
        resumo_estatisticas = df[colunas_numericas].describe()
        print(resumo_estatisticas)
        relatorio_validacao['estatisticas_numericas'] = resumo_estatisticas.to_dict()
    
    return relatorio_validacao

#Executa a limpeza
def executar_limpeza_completa():
    print("="*60)
    print("PROCESSO DE LIMPEZA E PRÉ-PROCESSAMENTO DE DADOS")
    print("="*60)

    dados_originais = carregar_dados()
    if dados_originais is None:
        return

    dados_processados = dados_originais.copy()
    colunas_redundantes = identificar_colunas_redundantes(dados_processados)
    inconsistencias = encontrar_inconcistencias(dados_processados)
    dados_processados, correcoes_aplicadas = corrigir_inconsistencias(dados_processados, inconsistencias)
    dados_processados, info_extremos = tratar_valores_extremos(dados_processados)
    dados_processados, info_em_falta = tratar_valores_em_falta(dados_processados, estrategia='auto')
    dados_processados, colunas_removidas = remover_colunas_desnecessarias(dados_processados, colunas_redundantes)
    relatorio_validacao = validar_dados_finais(dados_processados)

    try:
        dados_processados.to_csv(caminho_completo_saida, index=False, encoding='utf-8')
        print(f"\nDados limpos guardados em: {caminho_completo_saida}")
        print(f"Dimensões finais: {dados_processados.shape}")
    except Exception as erro:
        print(f"ERRO ao guardar dados limpos: {erro}")
    
    return dados_processados

def main():
    dados_limpos = executar_limpeza_completa()

if __name__ == "__main__":
    main()

