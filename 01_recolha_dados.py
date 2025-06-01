import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options


# Recolhe os links da PorData para os quais o utilizador deseja recolher os dados
def obter_links_do_usuario():
    try:
        quantidade = int(input("Introduza a quantidade de links: "))
        print("Introduza os links:")
        links = []
        while len(links) < quantidade:
            link = input(f"Link {len(links) + 1}/{quantidade}: ")
            if not link and len(links) < quantidade:
                print(
                    f"Ainda faltam {quantidade - len(links)} links. Continue a introduzir ou pressione Ctrl+C para cancelar.")
                continue
            elif not link and len(links) == quantidade:
                break
            links.append(link)

        print(f"Foram recolhidos {len(links)} links.")
        return links
    except ValueError:
        print("Erro: Por favor, introduz um número válido para a quantidade de links.")
        return obter_links_do_usuario()
    except KeyboardInterrupt:
        print("\nOperação cancelada pelo utilizador.")
        exit(1)


# Função para clicar no Elemento, com prevenção de erros
def click_elemento(driver, elemento, timeout=10, mensagem=None):
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elemento)
        time.sleep(1)
        try:
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, elemento.get_attribute("xpath"))))
            elemento.click()
            return True
        except (ElementClickInterceptedException, Exception):
            print("Click interceptado, a tentar alternativas...")

            try:
                ActionChains(driver).move_to_element(elemento).click().perform()
                return True
            except:
                pass

            try:
                driver.execute_script("arguments[0].click();", elemento)
                return True
            except:
                pass

            try:
                driver.execute_script("""
                    arguments[0].style.display = 'block';
                    arguments[0].style.visibility = 'visible';
                    arguments[0].click();
                """, elemento)
                return True
            except Exception as e:
                print(f"Todas as tentativas de click falharam: {str(e)}")
                return False
    except TimeoutException:
        if mensagem:
            print(f"Timeout ao tentar interagir com o elemento: {mensagem}")
        else:
            print(f"Timeout ao tentar interagir com o elemento")
        return False
    except Exception as e:
        if mensagem:
            print(f"Erro ao tentar clicar no elemento: {mensagem} - {str(e)}")
        else:
            print(f"Erro ao tentar clicar no elemento: {str(e)}")
        return False

def fechar_popups_cookies(driver):
    seletores_cookies = [
        (By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"),
        (By.ID, "CybotCookiebotDialogBodyButtonAccept"),
        (By.XPATH, "//div[@id='CybotCookiebotDialog']//button[contains(text(), 'Allow all')]"),
        (By.XPATH, "//div[@id='CybotCookiebotDialog']//button[contains(text(), 'Permitir todos')]"),
        (By.XPATH, "//div[@id='CybotCookiebotDialog']//button[contains(text(), 'Accept')]"),
        (By.XPATH, "//div[@id='CybotCookiebotDialog']//button[contains(text(), 'Aceitar')]"),
        (By.ID, "onetrust-accept-btn-handler"),
        (By.ID, "accept-cookies"),
        (By.ID, "acceptCookies"),
        (By.ID, "cookieConsent"),
        (By.XPATH, "//button[contains(text(), 'Accept')]"),
        (By.XPATH, "//button[contains(text(), 'Aceitar')]"),
        (By.XPATH, "//button[contains(text(), 'Allow all')]"),
        (By.XPATH, "//button[contains(text(), 'Permitir todos')]"),
        (By.XPATH, "//button[contains(@class, 'cookie') and contains(text(), 'Accept')]"),
        (By.XPATH, "//button[contains(@class, 'cookie') and contains(text(), 'Aceitar')]"),
        (By.XPATH, "//a[contains(text(), 'Accept') and contains(@class, 'cookie')]"),
        (By.XPATH, "//a[contains(text(), 'Aceitar') and contains(@class, 'cookie')]"),
        (By.XPATH, "//div[contains(@class, 'cookie')]//button"),
        (By.XPATH, "//div[contains(@id, 'cookie')]//button"),
        (By.XPATH, "//div[contains(@id, 'Cookie')]//button"),
    ]
    try:
        overlays = driver.find_elements(By.XPATH, "//div[contains(@id, 'CybotCookiebotDialogBodyUnderlay')]")
        for overlay in overlays:
            driver.execute_script("arguments[0].style.display = 'none';", overlay)

        overlays = driver.find_elements(By.XPATH, "//div[contains(@class, 'overlay') or contains(@class, 'backdrop')]")
        for overlay in overlays:
            driver.execute_script("arguments[0].style.display = 'none';", overlay)
    except:
        pass
    for by, valor in seletores_cookies:
        try:
            elemento = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((by, valor))
            )

            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elemento)
            time.sleep(0.5)
            try:
                elemento.click()
            except ElementClickInterceptedException:
                try:
                    driver.execute_script("arguments[0].click();", elemento)
                except:
                    ActionChains(driver).move_to_element(elemento).click().perform()

            time.sleep(2)
            return True

        except (TimeoutException, NoSuchElementException):
            continue

    return False


# Função que recolhe efetivamente os dados
def recolha_dados(nome, link, pasta_downloads):
    os.makedirs(pasta_downloads, exist_ok=True)

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    prefs = {
        "download.default_directory": pasta_downloads,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }

    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(link)
        print(f"Página carregada: {link}")
        time.sleep(2)
        fechar_popups_cookies(driver)
        time.sleep(3)
        selectors_download = [
            "//div[@class='download-csv']//a",
            "//a[contains(@class, 'download')]",
            "//button[contains(@class, 'download')]",
            "//a[contains(text(), 'CSV')]",
            "//button[contains(text(), 'CSV')]",
            "//a[contains(@href, 'csv')]"
        ]

        download_button = None
        for selector in selectors_download:
            try:
                download_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, selector))
                )
                if download_button.is_displayed():
                    break
            except TimeoutException:
                continue

        if not download_button:
            raise Exception("Botão de download não encontrado")

        print("Botão de download encontrado")

        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", download_button)
        time.sleep(2)
        try:
            download_button.click()
            print("Download iniciado com clique direto")
        except Exception:
            try:
                driver.execute_script("arguments[0].click();", download_button)
                print("Download iniciado com JavaScript")
            except Exception as e:
                print(f"Falha ao clicar no botão de download: {str(e)}")
                raise
        arquivo_csv = esperar_csv_novo(pasta_downloads)
        if not arquivo_csv:
            raise Exception("Timeout ao esperar pelo download do arquivo")

        novo_nome = os.path.join(pasta_downloads, f"{nome}.csv")
        if os.path.exists(novo_nome):
            os.remove(novo_nome)  # Remove se já existir
        os.rename(arquivo_csv, novo_nome)
        print(f"Ficheiro guardado como: {novo_nome}")

    except Exception as e:
        print(f"Erro durante a recolha de {nome}: {str(e)}")
        raise
    finally:
        driver.quit()


def esperar_csv_novo(pasta_downloads, timeout=60):
    arquivos_antes = set([os.path.join(pasta_downloads, arquivo) for arquivo in os.listdir(pasta_downloads)])
    extensoes = ['.csv', '.xlsx', '.xls']
    inicio = time.time()

    while time.time() - inicio < timeout:
        time.sleep(1)
        arquivos_atuais = set([os.path.join(pasta_downloads, arquivo) for arquivo in os.listdir(pasta_downloads)])
        novos_arquivos = arquivos_atuais - arquivos_antes

        for arquivo in novos_arquivos:
            _, extensao = os.path.splitext(arquivo.lower())
            if extensao in extensoes and not arquivo.endswith('.crdownload'):
                try:
                    with open(arquivo, 'rb') as f:
                        pass
                    return arquivo
                except:
                    continue

    return None


if __name__ == "__main__":
    pasta_downloads = os.path.join(os.path.abspath(os.path.dirname(__file__)), "Dados recolhidos")
    os.makedirs(pasta_downloads, exist_ok=True)
    links_usuario = obter_links_do_usuario()
    if links_usuario:
        for i, link in enumerate(links_usuario):
            nome = link.strip('/').split('/')[-1].replace('-', '_')
            try:
                recolha_dados(nome, link, pasta_downloads)
                print(f"Processo completo para o link {i + 1}")
            except Exception as e:
                print(f"Falha ao processar o link {i + 1}: {str(e)}")
    else:
        print("Nenhum link foi fornecido. O programa será encerrado.")
