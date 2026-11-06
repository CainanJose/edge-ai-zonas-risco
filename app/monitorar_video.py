
import argparse
import json
import time
from pathlib import Path

import cv2

from app.vision import analisar_imagem


def monitorar_video(
    fonte,
    cenario,
    saida,
    largura_max=640,
    intervalo=1,
    max_quadros=0
):
    """
    Processa um video ou camera quadro a quadro.

    Reutiliza o detector YOLO, as zonas e os alertas.
    """

    # Permite utilizar arquivos ou cameras (ex.: 0)
    origem = int(fonte) if fonte.isdecimal() else fonte

    captura = cv2.VideoCapture(origem)

    if not captura.isOpened():
        raise RuntimeError(
            f"Nao foi possivel abrir a fonte: {fonte}"
        )

    fps_original = captura.get(cv2.CAP_PROP_FPS)

    if fps_original <= 0:
        fps_original = 20.0

    fps_saida = fps_original / intervalo

    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)

    escritor = None

    quadros_lidos = 0
    quadros_processados = 0

    tempos = []

    contagem_alertas = {
        "NORMAL": 0,
        "ATENCAO": 0,
        "CRITICO": 0
    }

    inicio_total = time.perf_counter()

    try:
        while True:
            sucesso, imagem = captura.read()

            if not sucesso:
                break

            quadros_lidos += 1

            # Permite amostrar quadros para reduzir o custo
            if (quadros_lidos - 1) % intervalo != 0:
                continue

            # Reduz a resolucao sem distorcer a imagem
            altura, largura = imagem.shape[:2]

            if largura > largura_max:
                nova_altura = int(
                    altura * largura_max / largura
                )

                imagem = cv2.resize(
                    imagem,
                    (largura_max, nova_altura)
                )

            # Executa todo o pipeline existente
            inicio_quadro = time.perf_counter()

            dados, imagem_anotada = analisar_imagem(
                imagem,
                cenario
            )

            tempo_quadro = (
                time.perf_counter() - inicio_quadro
            ) * 1000

            tempos.append(tempo_quadro)

            nivel = dados["alerta"]["nivel"]
            contagem_alertas[nivel] += 1

            # Inicializa o arquivo de saida
            if escritor is None:
                h, w = imagem_anotada.shape[:2]

                codec = cv2.VideoWriter_fourcc(*"mp4v")

                escritor = cv2.VideoWriter(
                    str(saida),
                    codec,
                    fps_saida,
                    (w, h)
                )

                if not escritor.isOpened():
                    raise RuntimeError(
                        "Nao foi possivel criar o video MP4."
                    )

            escritor.write(imagem_anotada)

            quadros_processados += 1

            if quadros_processados % 10 == 0:
                print(
                    f"Quadro {quadros_processados} | "
                    f"Pessoas: {dados['total_pessoas']} | "
                    f"Alerta: {nivel} | "
                    f"Tempo: {tempo_quadro:.1f} ms"
                )

            if max_quadros > 0:
                if quadros_processados >= max_quadros:
                    break

    except KeyboardInterrupt:
        print("\nMonitoramento interrompido.")

    finally:
        captura.release()

        if escritor is not None:
            escritor.release()

    duracao = time.perf_counter() - inicio_total

    if quadros_processados == 0:
        raise RuntimeError(
            "Nenhum quadro foi processado."
        )

    resumo = {
        "cenario": cenario,
        "quadros_lidos": quadros_lidos,
        "quadros_processados": quadros_processados,
        "fps_origem": round(fps_original, 2),
        "fps_processamento": round(
            quadros_processados / duracao, 2
        ),
        "latencia_media_ms": round(
            sum(tempos) / len(tempos), 2
        ),
        "quadros_por_alerta": contagem_alertas,
        "tempo_total_s": round(duracao, 2),
        "video_saida": str(saida)
    }

    # Salva um relatorio das medicoes
    caminho_resumo = saida.with_suffix(".json")

    caminho_resumo.write_text(
        json.dumps(
            resumo,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== RESUMO DO MONITORAMENTO ===")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))

    return resumo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitoramento Edge AI por video"
    )

    parser.add_argument("--fonte", required=True)
    parser.add_argument(
        "--cenario",
        choices=["padrao", "deposito", "corredor"],
        default="deposito"
    )
    parser.add_argument(
        "--saida",
        default="outputs/monitoramento.mp4"
    )
    parser.add_argument(
        "--largura",
        type=int,
        default=640
    )
    parser.add_argument(
        "--intervalo",
        type=int,
        default=1
    )
    parser.add_argument(
        "--max-quadros",
        type=int,
        default=0
    )

    args = parser.parse_args()

    if args.intervalo < 1 or args.largura < 1:
        parser.error(
            "Intervalo e largura devem ser positivos."
        )

    monitorar_video(
        fonte=args.fonte,
        cenario=args.cenario,
        saida=args.saida,
        largura_max=args.largura,
        intervalo=args.intervalo,
        max_quadros=args.max_quadros
    )
