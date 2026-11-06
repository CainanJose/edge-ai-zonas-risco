
import argparse
import sys
import time
from pathlib import Path

import cv2

from app.vision import analisar_imagem
from app.zones import CENARIOS


def abrir_camera(indice_camera):
    """
    Abre uma camera utilizando backends compativeis.

    No Windows, prioriza Media Foundation (MSMF).
    Caso nao funcione, tenta DirectShow e CAP_ANY.
    """

    if sys.platform.startswith("win"):
        backends = [
            ("Media Foundation", cv2.CAP_MSMF),
            ("DirectShow", cv2.CAP_DSHOW),
            ("Automatico", cv2.CAP_ANY)
        ]
    else:
        backends = [
            ("Automatico", cv2.CAP_ANY)
        ]

    for nome, backend in backends:
        print(f"Tentando camera {indice_camera} com {nome}...")

        captura = cv2.VideoCapture(
            indice_camera,
            backend
        )

        if not captura.isOpened():
            captura.release()
            continue

        # Configuracao inicial da camera
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Tenta reduzir o buffer, se suportado
        captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Verifica se consegue capturar um quadro real
        sucesso, imagem = captura.read()

        if sucesso and imagem is not None:
            print(
                f"Camera {indice_camera} conectada "
                f"com sucesso usando {nome}!"
            )

            return captura

        captura.release()

    raise RuntimeError(
        f"Nao foi possivel abrir a camera {indice_camera}. "
        "Verifique o indice, a conexao USB e as "
        "permissoes do Windows."
    )


def monitorar_ao_vivo(
    fonte,
    cenario="padrao",
    largura_max=640,
    intervalo=1,
    repetir=False,
    max_quadros=0
):
    """
    Executa o monitoramento continuo utilizando
    webcam ou arquivo de video.

    Reutiliza o pipeline de deteccao, zonas e alertas.
    """

    # 1. Identifica se a fonte e webcam ou arquivo
    webcam = fonte.isdecimal()

    if webcam:
        indice_camera = int(fonte)

        captura = abrir_camera(indice_camera)

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

    # 2. Informacoes da fonte
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
            # 3. Captura o proximo quadro
            sucesso, imagem = captura.read()

            if not sucesso:
                if not webcam and repetir:
                    captura.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0
                    )
                    continue

                break

            lidos += 1

            # Amostragem opcional de quadros
            if (lidos - 1) % intervalo != 0:
                continue

            # 4. Redimensiona preservando a proporcao
            altura, largura = imagem.shape[:2]

            if largura > largura_max:
                nova_altura = int(
                    altura * largura_max / largura
                )

                imagem = cv2.resize(
                    imagem,
                    (largura_max, nova_altura),
                    interpolation=cv2.INTER_AREA
                )

            # 5. Executa o pipeline de visao computacional
            inicio_quadro = time.perf_counter()

            dados, anotada = analisar_imagem(
                imagem,
                cenario
            )

            tempo_ms = (
                time.perf_counter() - inicio_quadro
            ) * 1000

            tempos.append(tempo_ms)
            processados += 1

            # 6. Registra os alertas
            nivel = dados["alerta"]["nivel"]

            alertas[nivel] += 1

            # 7. Calcula o FPS efetivo
            duracao = (
                time.perf_counter() - inicio_total
            )

            fps_efetivo = processados / max(
                duracao,
                0.001
            )

            # 8. Adiciona informacoes no rodape
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

            # 9. Exibe o monitoramento
            cv2.imshow(
                "Edge AI - Monitoramento de Risco",
                anotada
            )

            # Webcam: exibe assim que o processamento termina
            # Video: tenta respeitar o FPS original
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

            # 10. Permite encerrar com Q ou ESC
            if tecla in (ord("q"), 27):
                break

            if max_quadros > 0:
                if processados >= max_quadros:
                    break

    finally:
        # Libera a webcam e fecha a janela
        captura.release()
        cv2.destroyAllWindows()

    # 11. Apresenta as metricas finais
    duracao_total = (
        time.perf_counter() - inicio_total
    )

    print("\n=== RESULTADO DO MONITORAMENTO ===")
    print(f"Fonte utilizada: {fonte}")
    print(f"Cenario: {cenario}")
    print(f"Quadros lidos: {lidos}")
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
        help="Indice da webcam (0, 1, 2) ou caminho do video"
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
