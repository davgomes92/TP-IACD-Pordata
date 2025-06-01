import pandas as pd
import unicodedata
from pathlib import Path
import re


# Classe para facilitar a integração já que todos os csv da PorData partilham carateristicas chave
class IntegradorPorData:
    def __init__(self, caminho_entrada="Dados recolhidos", caminho_saida="Dados integrados",
                 intervalo_anos=(2009, 2023)):
        self.caminho_entrada = Path(caminho_entrada)
        self.caminho_saida = Path(caminho_saida)
        self.ano_min, self.ano_max = intervalo_anos
        self.caminho_saida.mkdir(exist_ok=True)

        self.colunas = {
            'ano': '01. Ano',
            'regiao': '02. Nome Região (Portugal)',
            'ambito': '03. Âmbito Geográfico',
            'filtro1': '04. Filtro 1',
            'filtro2': '05. Filtro 2',
            'filtro3': '06. Filtro 3',
            'valor': '09. Valor'
        }

    def encontrar_csv(self):
        csvs = list(self.caminho_entrada.glob("*.csv"))
        print(f"Encontrados {len(csvs)} CSVs")
        return csvs

    def normalizar_texto(self, texto):
        texto = unicodedata.normalize('NFD', texto)
        texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
        return texto.lower().strip()

    def mapear_colunas(self, df_colunas):
        mapeamento = {}
        for col_padrao in self.colunas.values():
            if col_padrao in df_colunas:
                mapeamento[col_padrao] = col_padrao
                continue
            for df_col in df_colunas:
                if self.normalizar_texto(col_padrao) == self.normalizar_texto(df_col):
                    mapeamento[col_padrao] = df_col
                    break
        return mapeamento

    def limpar_nome_do_filtro(self, valor_de_filtro):
        if pd.isna(valor_de_filtro):
            return ""

        texto = str(valor_de_filtro)
        texto = re.sub(r'^\d+\.\s*', '', texto)
        texto = re.sub(r'[^\w\s]', '', texto)
        texto = texto.replace(' ', '_')
        return texto[:20]

    def criar_nome_metrica(self, nome_base, filtros):
        base = nome_base.replace('.csv', '')
        base = re.sub(r'[^\w]', '_', base)

        sufixos = []
        for f in filtros:
            if f and f != "Total":
                f_limpo = self.limpar_nome_do_filtro(f)
                if f_limpo:
                    sufixos.append(f_limpo)

        return f"{base}_{'_'.join(sufixos)}" if sufixos else base

    def processar_csv(self, caminho_csv):
        print(f"\nA processar: {caminho_csv.name}")

        try:
            df = pd.read_csv(caminho_csv, encoding='utf-8')
            print(f"  Registos carregados: {len(df)}")
            column_mapeamento = self.mapear_colunas(df.columns)
            col_necessarias = [self.colunas['ano'], self.colunas['regiao'], self.colunas['valor']]
            col_em_falta = [col for col in col_necessarias if col not in column_mapeamento]

            if col_em_falta:
                print(f"  ERRO: Colunas essenciais não encontradas: {col_em_falta}")
                return {}
            df_novo = df.rename(columns={v: k for k, v in column_mapeamento.items()})

            col_ano = self.colunas['ano']
            df_filtrado = df_novo[
                (df_novo[col_ano] >= self.ano_min) &
                (df_novo[col_ano] <= self.ano_max)
                ]

            print(f"  Registos após o filtro ({self.ano_min}-{self.ano_max}): {len(df_filtrado)}")

            if df_filtrado.empty:
                return {}

            return self.processar_por_filtros(df_filtrado, caminho_csv.name)

        except Exception as e:
            print(f"  ERRO: {e}")
            return {}

    def processar_por_filtros(self, df, filename):
        resultados = {}

        col_filtros = []
        for chave_filtro in ['filtro1', 'filtro2', 'filtro3']:
            col = self.colunas[chave_filtro]
            if col in df.columns:
                col_filtros.append(col)

        if not col_filtros:
            nome_da_metrica = self.criar_nome_metrica(filename, [])
            processados = self.criar_dataframe_metrica(df, nome_da_metrica)
            if not processados.empty:
                resultados[nome_da_metrica] = processados
            return resultados

        valores_unicos = {}
        for col in col_filtros:
            valores = df[col].unique()
            valores = [None if pd.isna(x) else x for x in valores]
            valores_unicos[col] = valores

        combinacoes_filtros = self.criar_combinacoes(col_filtros, valores_unicos)

        for combinacao in combinacoes_filtros:
            mascara = pd.Series([True] * len(df), index=df.index)

            for col, valor in zip(col_filtros, combinacao):
                if valor is None:
                    mascara &= df[col].isna()
                else:
                    mascara &= (df[col] == valor)

            df_subset = df[mascara]

            if not df_subset.empty:
                nome_da_metrica = self.criar_nome_metrica(filename, combinacao)
                processados = self.criar_dataframe_metrica(df_subset, nome_da_metrica)
                if not processados.empty:
                    resultados[nome_da_metrica] = processados
                    print(f"    {nome_da_metrica}: {len(processados)} registos")

        return resultados

    def criar_combinacoes(self, col_filtros, valores_unicos):
        combinacao = []
        if len(col_filtros) == 1:
            for v1 in valores_unicos[col_filtros[0]]:
                combinacao.append([v1])

        elif len(col_filtros) == 2:
            for v1 in valores_unicos[col_filtros[0]]:
                for v2 in valores_unicos[col_filtros[1]]:
                    combinacao.append([v1, v2])

        elif len(col_filtros) == 3:
            for v1 in valores_unicos[col_filtros[0]]:
                for v2 in valores_unicos[col_filtros[1]]:
                    for v3 in valores_unicos[col_filtros[2]]:
                        combinacao.append([v1, v2, v3])

        return combinacao

    def criar_dataframe_metrica(self, df, nome_da_metrica):
        colunas_amanter = [
            self.colunas['ano'],
            self.colunas['regiao'],
            self.colunas['valor']
        ]

        if self.colunas['ambito'] in df.columns:
            colunas_amanter.append(self.colunas['ambito'])

        df_result = df[colunas_amanter].copy()
        rename_map = {
            self.colunas['ano']: 'Ano',
            self.colunas['regiao']: 'Municipio',
            self.colunas['valor']: nome_da_metrica
        }

        if self.colunas['ambito'] in df_result.columns:
            rename_map[self.colunas['ambito']] = 'Ambito'

        df_result = df_result.rename(columns=rename_map)

        if 'Ambito' not in df_result.columns:
            df_result['Ambito'] = 'Municipal'

        df_result[nome_da_metrica] = pd.to_numeric(df_result[nome_da_metrica], errors='coerce')
        df_result = df_result.dropna(subset=[nome_da_metrica])

        return df_result

    def integrar_todos_dados(self):
        csvs = self.encontrar_csv()
        if not csvs:
            print("Nenhum CSV encontrado!")
            return pd.DataFrame()

        print(f"\nA integrar {len(csvs)} arquivos...")

        todos_os_dataframes = {}
        for caminho_csv in csvs:
            csv_resultados = self.processar_csv(caminho_csv)
            todos_os_dataframes.update(csv_resultados)

        if not todos_os_dataframes:
            print("Nenhum dado válido encontrado!")
            return pd.DataFrame()

        print(f"\nA integrar {len(todos_os_dataframes)} métricas...")

        df_final = None
        for nome_da_metrica, metrica_df in todos_os_dataframes.items():
            if df_final is None:
                df_final = metrica_df
            else:
                df_final = df_final.merge(
                    metrica_df,
                    on=['Ano', 'Municipio', 'Ambito'],
                    how='outer'
                )

        col_numericas = df_final.select_dtypes(include=['float64', 'int64']).columns
        col_numericas = [col for col in col_numericas if col != 'Ano']
        df_final[col_numericas] = df_final[col_numericas].fillna(0)

        df_final = df_final.sort_values(['Ano', 'Municipio'])

        print(f"\nIntegração concluída:")
        print(f"  Registos: {len(df_final)}")
        print(f"  Período: {df_final['Ano'].min()}-{df_final['Ano'].max()}")
        print(f"  Municípios: {df_final['Municipio'].nunique()}")
        print(f"  Métricas: {len(col_numericas)}")

        return df_final

    def guardar_dados_integrados(self, df, filename="dados_integrados_pordata.csv"):
        if df.empty:
            print("Nenhum dado para guardar!")
            return

        caminho_saida = self.caminho_saida / filename
        df.to_csv(caminho_saida, index=False, encoding='utf-8')
        print(f"\nDados guardados em: {caminho_saida}")

        print(f"\nColunas no arquivo final ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

    def executar_integracao(self, nome_do_arquivo="dados_integrados_pordata.csv"):
        print("=== INTEGRAÇÃO DE DADOS DA PORDATA ===")
        print(f"Entrada: {self.caminho_entrada}")
        print(f"Saída: {self.caminho_saida}")
        print(f"Período: {self.ano_min}-{self.ano_max}")

        df_integrada = self.integrar_todos_dados()

        if not df_integrada.empty:
            self.guardar_dados_integrados(df_integrada, nome_do_arquivo)

        return df_integrada

if __name__ == "__main__":
    integrador = IntegradorPorData(
        caminho_entrada="Dados recolhidos",
        caminho_saida="Dados integrados",
        intervalo_anos=(2009, 2023)
    )
    dados_integrados = integrador.executar_integracao()
    if not dados_integrados.empty:
        print(f"\nEstatísticas finais:")
        print(f"  Formato: {dados_integrados.shape}")