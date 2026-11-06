
import argparse
import sys
import time
from pathlib import Path

import cv2

from app.vision import analisar_imagem
from app.zones import CENARIOS


def monitorar_ao_vivo(
    fonte,
    cenario="padrao",
    largura_max=640,
    intervalo=1,
    repetir=False,
    max_quadros=0
):
    # Identifica se a fonte e webcam ou arquivo
    webcam = fonte.isdecimal()

    if webcam:
        indice_camera = int(fonte)

        if sys.platform.startswith("win"):
            captura = cv2.VideoCapture(
                indice_camera, cv2.CAP_DSHOW
            )
        else:
            captura = cv2.VideoCapture(indice_camera)

        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Tenta reduzir o buffer, se suportado
        captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    else:
        caminho = Path(fonte)

        if not caminho.is_file():
            raise FileNotFoundError(
                f"Video nao encontrado: {caminho}"
            )

        captura = cv2.VideoCapture(str(caminho))

    if not captura.isOpened():
        raise RuntimeError(
            f"Nao foi possivel abrir a fonte: {fonte}"
        )

    fps_origem = captura.get(cv2.CAP_PROP_FPS)

    if fps_origem <= 0:
        fps_origem = 10.0

    lidos = 0
    processados = 0
    tempos = []

    alertas = {
        "NORMAL": 0,
        "ATENCAO": 0,
        "CRITICO": 0
    }

    inicio_total = time.perf_counter()

    print("\n=== MONITORAMENTO EDGE AI ===")
    print(f"Fonte: {fonte}")
    print(f"Cenario: {cenario}")
    print("Pressione Q ou ESC para encerrar.\n")

    try:
        while True:
            sucesso, imagem = captura.read()

            if not sucesso:
                if not webcam and repetir:
                    captura.set(
                        cv2.CAP_PROP_POS_FRAMES, 0
                    )
                    continue

                break

            lidos += 1

            # Amostragem opcional de quadros
            if (lidos - 1) % intervalo != 0:
                continue

            altura, largura = imagem.shape[:2]

            # Redimensiona mantendo a proporcao
            if largura > largura_max:
                nova_altura = int(
                    altura * largura_max / largura
                )

                imagem = cv2.resize(
                    imagem,
                    (largura_max, nova_altura),
                    interpolation=cv2.INTER_AREA
                )

            # Reutiliza o pipeline completo
            inicio_quadro = time.perf_counter()

            dados, anotada = analisar_imagem(
                imagem, cenario
            )

            tempo_ms = (
                time.perf_counter() - inicio_quadro
            ) * 1000

            tempos.append(tempo_ms)
            processados += 1

            nivel = dados["alerta"]["nivel"]
            alertas[nivel] += 1

            duracao = (
                time.perf_counter() - inicio_total
            )

            fps_efetivo = processados / max(
                duracao, 0.001
            )

            # Mostra informacoes no rodape
            h, w = anotada.shape[:2]

            cv2.rectangle(
                anotada,
                (0, max(0, h - 34)),
                (w - 1, h - 1),
                (20, 20, 20),
                -1
            )

            texto = (
                f"FPS: {fps_efetivo:.1f} | "
                f"Latencia: {tempo_ms:.0f} ms | "
                f"Pessoas: {dados['total_pessoas']}"
            )

            cv2.putText(
                anotada,
                texto,
                (10, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Edge AI - Monitoramento de Risco",
                anotada
            )

            # Webcam: apresenta o quadro assim que pronto.
            # Video: tenta respeitar o FPS original.
            espera_ms = 1

            if not webcam:
                periodo_ms = (
                    1000 * intervalo / fps_origem
                )

                espera_ms = max(
                    1,
                    int(periodo_ms - tempo_ms)
                )

            tecla = cv2.waitKey(espera_ms) & 0xFF

            if tecla in (ord("q"), 27):
                break

            if max_quadros > 0:
                if processados >= max_quadros:
                    break

    finally:
        captura.release()
        cv2.destroyAllWindows()

    duracao_total = (
        time.perf_counter() - inicio_total
    )

    print("\n=== RESULTADO ===")
    print(f"Quadros processados: {processados}")

    if processados > 0:
        print(
            f"FPS efetivo: "
            f"{processados / duracao_total:.2f}"
        )

        print(
            f"Latencia media: "
            f"{sum(tempos) / len(tempos):.2f} ms"
        )

    print(f"Alertas por quadro: {alertas}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitoramento ao vivo com Edge AI"
    )

    parser.add_argument(
        "--fonte",
        required=True,
        help="Indice da webcam (0) ou caminho de video"
    )

    parser.add_argument(
        "--cenario",
        choices=list(CENARIOS.keys()),
        default="padrao"
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
        "--repetir",
        action="store_true"
    )

    parser.add_argument(
        "--max-quadros",
        type=int,
        default=0
    )

    args = parser.parse_args()

    if args.largura < 1 or args.intervalo < 1:
        parser.error(
            "Largura e intervalo devem ser positivos."
        )

    monitorar_ao_vivo(
        fonte=args.fonte,
        cenario=args.cenario,
        largura_max=args.largura,
        intervalo=args.intervalo,
        repetir=args.repetir,
        max_quadros=args.max_quadros
    )
