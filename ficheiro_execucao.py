import subprocess
import sys
import time
from pathlib import Path

def executar_script(script_name):
    print(f"\n=== A executar {script_name} ===")
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"✓ {script_name} concluído")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Erro em {script_name}: {e}")
        return False

def main():
    scripts = [
        "01_recolha_dados.py",
        "02_integracao_dados.py",
        "03_analise_exploratoria.py",
        "04_limpeza_preprocessamento.py",
        "05_analise_descritiva.py"
    ]
    print("PROCESSO PORDATA - INÍCIO")

    Path("Dados recolhidos").mkdir(exist_ok=True)
    Path("Dados integrados").mkdir(exist_ok=True)

    sucessos = 0
    for script in scripts:
        if executar_script(script):
            sucessos += 1
        time.sleep(4)
    print(f"\n=== CONCLUÍDO ===")
    print(f"Scripts executados: {sucessos}/{len(scripts)}")
    
    return sucessos == len(scripts)


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrompido pelo utilizador")
        sys.exit(1)