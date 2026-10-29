import ctypes
import glob
import importlib.util
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

MODELOS = {
    "Rápido (small)": "small",
    "Médio (medium)": "medium",
    "Alta qualidade (large-v3)": "large-v3",
}
VIDEOS = "*.mp4 *.mkv *.mov *.webm *.avi *.flv *.wmv *.m4v"
AUDIOS = "*.wav *.mp3 *.m4a *.aac *.ogg *.opus *.flac *.wma"
CREATE_NO_WINDOW = 0x08000000
PASTA_APP = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_LOG = os.path.join(PASTA_APP, "transcritor.log")

for _fluxo in ("stdout", "stderr"):
    # Sem console (pythonw) o Python deixa stdout/stderr como None e as barras de progresso
    # das bibliotecas (download do modelo) quebram ao escrever neles.
    if getattr(sys, _fluxo) is None:
        setattr(sys, _fluxo, open(os.devnull, "w", encoding="utf-8"))

logging.basicConfig(filename=ARQUIVO_LOG, level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transcritor")
FILTRO_MELHORAR = "highpass=f=80,loudnorm"


def formatar_tempo_srt(segundos):
    ms = int(round(segundos * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def formatar_tempo(segundos):
    m, s = divmod(int(segundos), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"


def duracao(arquivo):
    saida = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", arquivo],
        capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
    return float(saida.stdout.strip() or 0)


def extrair_audio(origem, destino, melhorar, ao_progresso=None):
    """Extrai WAV 16 kHz mono PCM 16 bits, o formato nativo do Whisper."""
    comando = ["ffmpeg", "-y", "-nostdin", "-i", origem, "-vn", "-ac", "1", "-ar", "16000"]
    if melhorar:
        comando += ["-af", FILTRO_MELHORAR]
    comando += ["-c:a", "pcm_s16le", "-progress", "pipe:1", "-nostats", destino]
    total = duracao(origem)
    proc = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, creationflags=CREATE_NO_WINDOW)
    for linha in proc.stdout:
        if linha.startswith("out_time_us=") and total and ao_progresso:
            try:
                ao_progresso(min(99, int(linha.split("=")[1]) / 1_000_000 / total * 100))
            except ValueError:
                pass
    erro = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(erro.strip().splitlines()[-1] if erro.strip() else "Falha no ffmpeg.")


def registrar_dlls_cuda():
    # As DLLs do CUDA (cublas, cudnn) vêm junto do PyTorch; o ctranslate2 não as acha sozinho
    spec = importlib.util.find_spec("torch")
    if spec and spec.submodule_search_locations:
        pasta = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        if os.path.isdir(pasta):
            os.add_dll_directory(pasta)
            os.environ["PATH"] = pasta + os.pathsep + os.environ["PATH"]


def carregar_modelo(nome, cpu=False):
    registrar_dlls_cuda()
    from faster_whisper import WhisperModel
    if not cpu:
        for tipo in ("float16", "int8_float16"):
            try:
                log.info("Carregando %s em cuda/%s", nome, tipo)
                return WhisperModel(nome, device="cuda", compute_type=tipo)
            except Exception:
                log.exception("Falha ao carregar %s em cuda/%s", nome, tipo)
    log.info("Carregando %s em cpu/int8", nome)
    return WhisperModel(nome, device="cpu", compute_type="int8")


def transcrever(nome, entrada, ao_trecho):
    """Transcreve na GPU; se a GPU falhar no meio do caminho, refaz na CPU."""
    for cpu in (False, True):
        modelo = carregar_modelo(nome, cpu)
        try:
            segmentos, info = modelo.transcribe(
                entrada, language="pt", beam_size=5, vad_filter=True,
                condition_on_previous_text=False)
            return [ao_trecho(seg, info.duration) for seg in segmentos]
        except Exception:
            log.exception("Falha ao transcrever (%s)", "cpu" if cpu else "cuda")
            if cpu:
                raise
            ao_trecho(None, 0)  # avisa a tela que vai tentar de novo na CPU


def diarizar(wav16k, pessoas, token):
    """Devolve [(inicio, fim, 'Pessoa N')] a partir de um WAV 16 kHz mono."""
    import gc
    import wave

    import numpy as np
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # libera a memória da placa que o Whisper usou
    registrar_dlls_cuda()
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # modelos do pyannote são checkpoints confiáveis
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    with wave.open(wav16k, "rb") as w:
        dados = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        taxa = w.getframerate()
    audio = {"waveform": torch.from_numpy(dados).unsqueeze(0), "sample_rate": taxa}
    resultado = pipeline(audio, num_speakers=pessoas or None)
    nomes = {}
    falas = []
    # "exclusive": um falante por vez, o que casa melhor com os trechos do Whisper
    for turno, _, rotulo in resultado.exclusive_speaker_diarization.itertracks(yield_label=True):
        nomes.setdefault(rotulo, f"Pessoa {len(nomes) + 1}")
        falas.append((turno.start, turno.end, nomes[rotulo]))
    return falas


def pessoa_do_trecho(inicio, fim, falas):
    """Quem fala mais durante o trecho (ou a fala mais próxima, se não houver sobreposição)."""
    tempos = {}
    for i, f, quem in falas:
        sobra = min(fim, f) - max(inicio, i)
        if sobra > 0:
            tempos[quem] = tempos.get(quem, 0) + sobra
    if tempos:
        return max(tempos, key=tempos.get)
    meio = (inicio + fim) / 2
    return min(falas, key=lambda x: min(abs(x[0] - meio), abs(x[1] - meio)))[2] if falas else ""


def salvar_token(token):
    os.environ["HF_TOKEN"] = token
    subprocess.run(["setx", "HF_TOKEN", token], check=True, capture_output=True,
                   creationflags=CREATE_NO_WINDOW)


def localizar_codex():
    """O 'codex' do PATH pode ser uma CLI antiga; prefere a mais recente instalada pelo app."""
    pasta = os.path.join(os.environ.get("LOCALAPPDATA", ""), "OpenAI", "Codex", "bin")
    candidatos = glob.glob(os.path.join(pasta, "*", "codex.exe"))
    return max(candidatos, key=os.path.getmtime) if candidatos else "codex"


ESTILO_PDF = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #1f2937; }
h1 { font-size: 20pt; border-bottom: 2px solid #2563eb; padding-bottom: 6px; }
h2 { font-size: 14pt; color: #1d4ed8; margin-top: 22px; }
h3 { font-size: 12pt; }
li { margin: 3px 0; }
hr { border: 0; border-top: 1px solid #d1d5db; }
"""


def localizar_navegador():
    """Edge (vem no Windows) ou Chrome, para imprimir o HTML em PDF sem instalar nada."""
    bases = [os.environ.get("ProgramFiles(x86)", ""), os.environ.get("ProgramFiles", ""),
             os.environ.get("LOCALAPPDATA", "")]
    relativos = [r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"]
    for rel in relativos:
        for base in bases:
            caminho = os.path.join(base, rel)
            if base and os.path.exists(caminho):
                return caminho
    return None


def markdown_para_pdf(markdown, destino):
    from markdown_it import MarkdownIt

    navegador = localizar_navegador()
    if not navegador:
        raise RuntimeError("Edge/Chrome não encontrado para gerar o PDF.")
    html = ("<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<style>{ESTILO_PDF}</style></head><body>{MarkdownIt().render(markdown)}</body></html>")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as pasta:
        arquivo_html = os.path.join(pasta, "revisao.html")
        with open(arquivo_html, "w", encoding="utf-8") as f:
            f.write(html)
        subprocess.run(
            [navegador, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={os.path.join(pasta, 'perfil')}",
             f"--print-to-pdf={destino}", "file:///" + arquivo_html.replace("\\", "/")],
            check=True, capture_output=True, timeout=120, creationflags=CREATE_NO_WINDOW)
        # o navegador pode terminar de gravar o PDF depois de encerrar: espera ainda com o HTML e o perfil no lugar
        for _ in range(40):
            if os.path.exists(destino) and os.path.getsize(destino) > 1000:
                return
            time.sleep(0.5)
    raise RuntimeError("O PDF não foi gerado.")


def revisar_com_ia(transcricao):
    """Pede a revisão ao Claude; se falhar (limite, erro, falta de acesso), tenta o Codex.

    Devolve (markdown, nome da ferramenta). Se nenhuma funcionar, levanta RuntimeError
    com o motivo de cada tentativa.
    """
    with open(os.path.join(PASTA_APP, "prompt_revisao.txt"), encoding="utf-8") as f:
        instrucoes = f.read()
    neutro = tempfile.gettempdir()  # fora de projetos, para a IA não carregar contexto de outros lugares
    saida_codex = os.path.join(neutro, "transcritor_codex_saida.txt")

    tentativas = [
        ("Claude", ["claude", "-p", instrucoes, "--tools", "", "--no-session-persistence",
                    "--setting-sources", "", "--disable-slash-commands"], None),
        ("Codex", [localizar_codex(), "exec", "--skip-git-repo-check", "-s", "read-only",
                   "-o", saida_codex, instrucoes], saida_codex),
    ]
    motivos = []
    for nome, comando, arquivo_saida in tentativas:
        try:
            if arquivo_saida and os.path.exists(arquivo_saida):
                os.remove(arquivo_saida)
            log.info("Revisão: tentando %s", nome)
            r = subprocess.run(comando, input=transcricao, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", cwd=neutro, timeout=900,
                               creationflags=CREATE_NO_WINDOW)
            texto = r.stdout
            if arquivo_saida:
                texto = open(arquivo_saida, encoding="utf-8").read() if os.path.exists(arquivo_saida) else ""
            texto = texto.strip()
            if texto.startswith("```"):  # alguns modelos embrulham o documento em um bloco de código
                texto = texto.split("\n", 1)[1] if "\n" in texto else ""
                texto = texto.rsplit("```", 1)[0].strip()
            if r.returncode == 0 and len(texto) > 200:
                return texto, nome
            detalhe = (r.stderr or r.stdout or "").strip()[-300:]
            motivos.append(f"{nome}: código {r.returncode}, resposta de {len(texto)} caracteres. {detalhe}")
        except Exception as erro:
            motivos.append(f"{nome}: {type(erro).__name__}: {erro}")
        log.error("Revisão falhou: %s", motivos[-1])
    raise RuntimeError(" | ".join(motivos))


class JanelaNomes(tk.Toplevel):
    """Mostra alguns trechos de cada 'Pessoa N' e pergunta o nome; nome vazio mantém 'Pessoa N'."""

    def __init__(self, mestre, resumo, resultado, pronto):
        super().__init__(mestre)
        self.title("Quem é cada pessoa?")
        self.geometry("640x520")
        self.transient(mestre)
        self.resultado = resultado
        self.pronto = pronto
        self.campos = {}
        self.protocol("WM_DELETE_WINDOW", self.pular)

        ttk.Label(self, text="Veja alguns trechos de cada voz e digite o nome (deixe em branco para manter).",
                  padding=10).pack(anchor="w")
        rodape = ttk.Frame(self, padding=10)
        rodape.pack(side="bottom", fill="x")
        ttk.Button(rodape, text="Aplicar nomes", command=self.aplicar).pack(side="right", padx=5)
        ttk.Button(rodape, text="Pular", command=self.pular).pack(side="right")

        area = ttk.Frame(self)
        area.pack(fill="both", expand=True, padx=10)
        tela = tk.Canvas(area, highlightthickness=0)
        barra = ttk.Scrollbar(area, orient="vertical", command=tela.yview)
        conteudo = ttk.Frame(tela)
        conteudo.bind("<Configure>", lambda _: tela.configure(scrollregion=tela.bbox("all")))
        tela.create_window((0, 0), window=conteudo, anchor="nw", width=590)
        tela.configure(yscrollcommand=barra.set)
        tela.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")
        tela.bind_all("<MouseWheel>", lambda e: tela.yview_scroll(int(-e.delta / 120), "units"))

        for rotulo, (segundos, exemplos) in resumo.items():
            caixa = ttk.LabelFrame(conteudo, text=f"{rotulo} — fala por {formatar_tempo(segundos)}", padding=8)
            caixa.pack(fill="x", pady=5)
            for inicio, texto in exemplos:
                ttk.Label(caixa, text=f"[{formatar_tempo(inicio)}] {texto}", wraplength=560,
                          justify="left").pack(anchor="w")
            linha = ttk.Frame(caixa)
            linha.pack(fill="x", pady=(6, 0))
            ttk.Label(linha, text="Nome:").pack(side="left")
            campo = ttk.Entry(linha)
            campo.pack(side="left", fill="x", expand=True, padx=5)
            self.campos[rotulo] = campo
        if self.campos:
            next(iter(self.campos.values())).focus_set()
        self.grab_set()

    def aplicar(self):
        for rotulo, campo in self.campos.items():
            nome = campo.get().strip()
            if nome:
                self.resultado[rotulo] = nome
        self.fechar()

    def pular(self):
        self.fechar()

    def fechar(self):
        self.unbind_all("<MouseWheel>")
        self.grab_release()
        self.destroy()
        self.pronto.set()


def resumir_pessoas(trechos, quem, exemplos_por_pessoa=4, limite=130):
    """{'Pessoa 1': (segundos falados, [(inicio, texto)...])}, com os trechos mais longos de cada voz."""
    resumo = {}
    for (inicio, fim, texto), pessoa in zip(trechos, quem):
        if not pessoa:
            continue
        segundos, lista = resumo.setdefault(pessoa, [0.0, []])
        resumo[pessoa][0] = segundos + (fim - inicio)
        lista.append((inicio, texto))
    final = {}
    for pessoa in sorted(resumo, key=lambda p: int(p.rsplit(" ", 1)[1]) if p.rsplit(" ", 1)[-1].isdigit() else 0):
        segundos, lista = resumo[pessoa]
        maiores = sorted(sorted(lista, key=lambda x: len(x[1]), reverse=True)[:exemplos_por_pessoa])
        final[pessoa] = (segundos, [(i, t if len(t) <= limite else t[:limite] + "…") for i, t in maiores])
    return final


class Aba(ttk.Frame):
    """Base: linha de arquivo, linha de pasta de saída, barra e status."""

    def __init__(self, mestre, tipos_entrada, titulo_entrada):
        super().__init__(mestre, padding=10)
        self.tipos_entrada = tipos_entrada
        self.titulo_entrada = titulo_entrada
        self.arquivo = tk.StringVar()
        self.pasta = tk.StringVar()
        self.status = tk.StringVar(value="Escolha um arquivo.")
        self.pasta_manual = False
        self.columnconfigure(1, weight=1)

        pad = {"padx": 8, "pady": 5}
        ttk.Label(self, text="Arquivo:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.arquivo, state="readonly").grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(self, text="Procurar...", command=self.procurar).grid(row=0, column=2, **pad)

        ttk.Label(self, text="Salvar em:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.pasta, state="readonly").grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(self, text="Alterar...", command=self.alterar_pasta).grid(row=1, column=2, **pad)

    def montar_rodape(self, linha, texto_botao):
        pad = {"padx": 8, "pady": 5}
        self.botao = ttk.Button(self, text=texto_botao, command=self.iniciar, state="disabled")
        self.botao.grid(row=linha, column=0, columnspan=3, pady=(15, 5))
        self.barra = ttk.Progressbar(self, maximum=100)
        self.barra.grid(row=linha + 1, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(self, textvariable=self.status).grid(row=linha + 2, column=0, columnspan=3, sticky="w", **pad)
        self.acoes = ttk.Frame(self)
        self.acoes.grid(row=linha + 3, column=0, columnspan=3)
        ttk.Button(self.acoes, text="Abrir pasta", command=lambda: os.startfile(self.pasta.get())).pack(side="left", padx=5)
        self.extras_acoes(self.acoes)
        self.acoes.grid_remove()

    def nome_base(self):
        return os.path.splitext(os.path.basename(self.arquivo.get()))[0]

    def pasta_saida(self):
        """Pasta com o nome do arquivo original, dentro de 'Salvar em' (sem aninhar duas vezes)."""
        raiz = self.pasta.get()
        base = self.nome_base()
        return raiz if os.path.basename(raiz) == base else os.path.join(raiz, base)

    def extras_acoes(self, frame):
        """Gancho para as abas acrescentarem botões ao terminar."""

    def garantir_token(self):
        """Pede o token do Hugging Face (janela) se ainda não estiver salvo. Roda na thread da tela."""
        if os.environ.get("HF_TOKEN"):
            return True
        token = simpledialog.askstring(
            "Token do Hugging Face",
            "Cole o token (começa com hf_). Ele fica salvo só neste computador,\n"
            "na variável de ambiente HF_TOKEN.", show="*", parent=self)
        if not token:
            return False
        token = token.strip()
        if not token.startswith("hf_"):
            messagebox.showwarning("Transcritor", "Token inválido: ele deve começar com hf_.")
            return False
        try:
            salvar_token(token)
        except Exception as erro:
            log.exception("Falha ao salvar o token")
            messagebox.showerror("Transcritor", f"Não consegui salvar o token: {erro}")
            return False
        return True

    def nomear_pessoas(self, trechos, quem):
        """Abre a janela de nomes (a partir da thread de trabalho) e devolve 'quem' já com os nomes."""
        self.atualizar("Aguardando você dar nome às pessoas...", 99)
        nomes = {}
        pronto = threading.Event()
        resumo = resumir_pessoas(trechos, quem)
        self.after(0, lambda: JanelaNomes(self.winfo_toplevel(), resumo, nomes, pronto))
        pronto.wait()
        log.info("Nomes das pessoas: %s", nomes or "nenhum informado")
        return [nomes.get(p, p) for p in quem], bool(nomes)

    def revisar_e_gerar_pdf(self, caminho_base, texto):
        """Revisão com IA (Claude, depois Codex) e PDF. Devolve o trecho de aviso para o status."""
        self.atualizar("Gerando a revisão com IA (pode levar alguns minutos)...", 99)
        try:
            markdown, ferramenta = revisar_com_ia(texto)
            self.salvar_revisao(markdown, caminho_base)
            return f" · revisão feita pelo {ferramenta}"
        except Exception:
            log.exception("Falha ao gerar a revisão com IA")
            mensagem = ("Não consegui gerar a revisão com IA (Claude e Codex falharam, talvez por limite de uso).\n\n"
                        "Os arquivos de texto foram salvos. Tente de novo depois pela aba 3.\n\n"
                        f"Detalhes no log: {ARQUIVO_LOG}")
            self.after(0, lambda: messagebox.showwarning("Transcritor", mensagem))
            return " · sem revisão com IA"

    def salvar_revisao(self, markdown, caminho_base):
        try:
            markdown_para_pdf(markdown, caminho_base + " - revisado.pdf")
        except Exception:
            log.exception("Falha ao gerar o PDF; salvando a revisão em .md")
            with open(caminho_base + " - revisado.md", "w", encoding="utf-8") as f:
                f.write(markdown)

    def definir_arquivo(self, caminho):
        caminho = os.path.normpath(caminho)
        self.arquivo.set(caminho)
        if not self.pasta_manual:
            self.pasta.set(os.path.dirname(caminho))
        self.botao.config(state="normal")
        self.acoes.grid_remove()
        self.barra["value"] = 0
        self.status.set("Pronto.")

    def procurar(self):
        caminho = filedialog.askopenfilename(
            title=self.titulo_entrada,
            filetypes=[(self.titulo_entrada, self.tipos_entrada), ("Todos os arquivos", "*.*")])
        if caminho:
            self.definir_arquivo(caminho)

    def alterar_pasta(self):
        pasta = filedialog.askdirectory(title="Pasta de saída", initialdir=self.pasta.get() or None)
        if pasta:
            self.pasta.set(os.path.normpath(pasta))
            self.pasta_manual = True

    def atualizar(self, status, pct=None):
        def aplicar():
            self.status.set(status)
            if pct is not None:
                self.barra["value"] = pct
        self.after(0, aplicar)

    def iniciar(self):
        if not self.validar():
            return
        self.botao.config(state="disabled")
        self.acoes.grid_remove()
        self.barra["value"] = 0
        threading.Thread(target=self.rodar, daemon=True).start()

    def validar(self):
        return True

    def rodar(self):
        try:
            self.executar()
        except Exception as erro:
            log.exception("Erro na operação")
            mensagem = f"{type(erro).__name__}: {erro}\n\nDetalhes no log:\n{ARQUIVO_LOG}"
            self.atualizar("Erro. Veja o log para os detalhes.")
            self.after(0, lambda: messagebox.showerror("Transcritor", mensagem))
        finally:
            self.after(0, lambda: self.botao.config(state="normal"))


class AbaExtrair(Aba):
    def __init__(self, mestre, ao_concluir):
        super().__init__(mestre, VIDEOS, "Vídeos")
        self.ao_concluir = ao_concluir
        self.destino = None
        self.melhorar = tk.BooleanVar(value=False)
        ttk.Label(self, text="Gera um WAV 16 kHz mono (o formato nativo do Whisper).").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=8, pady=5)
        ttk.Checkbutton(self, text="Melhorar áudio (corta graves e normaliza o volume)",
                        variable=self.melhorar).grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=5)
        self.montar_rodape(4, "Extrair áudio")

    def extras_acoes(self, frame):
        ttk.Button(frame, text="Transcrever este áudio →",
                   command=lambda: self.ao_concluir(self.destino)).pack(side="left", padx=5)

    def validar(self):
        self.destino = os.path.join(self.pasta_saida(), self.nome_base() + ".wav")
        if os.path.exists(self.destino):
            return messagebox.askyesno("Transcritor", f"{os.path.basename(self.destino)} já existe. Substituir?")
        return True

    def executar(self):
        log.info("Extraindo áudio de %s para %s", self.arquivo.get(), self.destino)
        self.atualizar("Extraindo o áudio... 0%", 0)
        os.makedirs(os.path.dirname(self.destino), exist_ok=True)
        extrair_audio(self.arquivo.get(), self.destino, self.melhorar.get(),
                      lambda pct: self.atualizar(f"Extraindo o áudio... {pct:.0f}%", pct))
        self.atualizar(f"Concluído: {os.path.basename(self.destino)}", 100)
        self.after(0, self.acoes.grid)


class AbaTranscrever(Aba):
    def __init__(self, mestre):
        super().__init__(mestre, AUDIOS + " " + VIDEOS, "Áudio ou vídeo")
        self.modelo = tk.StringVar(value="Alta qualidade (large-v3)")
        self.gerar_txt = tk.BooleanVar(value=True)
        self.gerar_srt = tk.BooleanVar(value=True)
        self.melhorar = tk.BooleanVar(value=False)
        self.separar = tk.BooleanVar(value=False)
        self.pessoas = tk.IntVar(value=0)
        self.revisar = tk.BooleanVar(value=True)

        pad = {"padx": 8, "pady": 5}
        ttk.Label(self, text="Modelo:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.modelo, values=list(MODELOS), state="readonly").grid(
            row=2, column=1, sticky="ew", **pad)
        opcoes = ttk.Frame(self)
        opcoes.grid(row=3, column=0, columnspan=3, sticky="w", **pad)
        ttk.Checkbutton(opcoes, text="Gerar .txt", variable=self.gerar_txt).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(opcoes, text="Gerar .srt (com tempos)", variable=self.gerar_srt).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(opcoes, text="Melhorar áudio", variable=self.melhorar).pack(side="left")
        vozes = ttk.Frame(self)
        vozes.grid(row=4, column=0, columnspan=3, sticky="w", **pad)
        ttk.Checkbutton(vozes, text="Separar por quem fala", variable=self.separar).pack(side="left", padx=(0, 15))
        ttk.Label(vozes, text="Nº de pessoas (0 = automático):").pack(side="left")
        ttk.Spinbox(vozes, from_=0, to=10, width=4, textvariable=self.pessoas).pack(side="left", padx=5)
        ttk.Checkbutton(self, text="Gerar revisão com IA (resumo, tarefas e texto organizado; usa Claude e, se falhar, Codex)",
                        variable=self.revisar).grid(row=5, column=0, columnspan=3, sticky="w", **pad)
        self.montar_rodape(6, "Transcrever")

    def validar(self):
        if not (self.gerar_txt.get() or self.gerar_srt.get()):
            messagebox.showwarning("Transcritor", "Marque pelo menos um formato de saída.")
            return False
        return not self.separar.get() or self.garantir_token()

    def executar(self):
        temporario = None
        try:
            origem = self.arquivo.get()
            pasta_destino = self.pasta_saida()
            os.makedirs(pasta_destino, exist_ok=True)
            base = os.path.join(pasta_destino, self.nome_base())
            entrada = origem
            if self.melhorar.get():
                self.atualizar("Preparando o áudio...")
                temporario = os.path.join(tempfile.gettempdir(), "transcritor_preparado.wav")
                extrair_audio(origem, temporario, True)
                entrada = temporario

            nome = MODELOS[self.modelo.get()]
            self.atualizar(f"Carregando o modelo {nome} (na primeira vez pode baixar)...")

            def ao_trecho(seg, total):
                if seg is None:
                    self.atualizar("A placa de vídeo falhou; refazendo na CPU (mais lento)...", 0)
                    return None
                pct = min(99, seg.end / (total or 1) * 100)
                self.atualizar(f"Transcrevendo... {pct:.0f}%  ({formatar_tempo(seg.end)} de {formatar_tempo(total)})", pct)
                return (seg.start, seg.end, seg.text.strip())

            log.info("Transcrevendo %s com %s", origem, nome)
            trechos = transcrever(nome, entrada, ao_trecho)
            log.info("Transcrição concluída: %d trechos", len(trechos))

            aviso = ""
            quem = [""] * len(trechos)
            if self.separar.get():
                try:
                    self.atualizar("Identificando quem fala (pode demorar alguns minutos)...", 99)
                    wav16k = entrada
                    if not self.melhorar.get():
                        temporario = os.path.join(tempfile.gettempdir(), "transcritor_preparado.wav")
                        extrair_audio(origem, temporario, False)
                        wav16k = temporario
                    log.info("Separando vozes (pessoas=%s)", self.pessoas.get())
                    falas = diarizar(wav16k, self.pessoas.get(), os.environ["HF_TOKEN"])
                    quem = [pessoa_do_trecho(i, f, falas) for i, f, _ in trechos]
                    log.info("Vozes separadas: %d pessoas", len({p for _, _, p in falas}))
                except Exception:
                    log.exception("Falha ao separar por quem fala")
                    aviso = " (sem separar vozes: veja o log)"

            if any(quem):
                quem, _ = self.nomear_pessoas(trechos, quem)

            def com_pessoa(n, t):
                return f"{quem[n]}: {t}" if quem[n] else t

            texto = "\n".join(f"[{formatar_tempo(i)}] {com_pessoa(n, t)}" for n, (i, _, t) in enumerate(trechos))
            if self.gerar_txt.get():
                with open(base + ".txt", "w", encoding="utf-8") as f:
                    f.write(texto)
            if self.gerar_srt.get():
                with open(base + ".srt", "w", encoding="utf-8") as f:
                    for n, (i, fim, t) in enumerate(trechos):
                        f.write(f"{n + 1}\n{formatar_tempo_srt(i)} --> {formatar_tempo_srt(fim)}\n{com_pessoa(n, t)}\n\n")

            if self.revisar.get():
                aviso += self.revisar_e_gerar_pdf(base, texto)

            self.atualizar("Concluído: 100%" + aviso, 100)
            self.after(0, self.acoes.grid)
        finally:
            if temporario and os.path.exists(temporario):
                os.remove(temporario)


def ler_trechos(pasta, base, arquivo):
    """Lê a transcrição (.srt tem início e fim; .txt só o início). Devolve (trechos, rótulos 'Pessoa N')."""
    srt = os.path.join(pasta, base + ".srt")
    txt = os.path.join(pasta, base + ".txt")
    origem = srt if os.path.exists(srt) else (arquivo if arquivo.lower().endswith((".srt", ".txt")) else txt)
    with open(origem, encoding="utf-8") as f:
        conteudo = f.read()
    brutos = []  # (inicio, fim ou None, texto)
    if origem.lower().endswith(".srt"):
        for bloco in re.split(r"\n\s*\n", conteudo.strip()):
            linhas = bloco.strip().splitlines()
            m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", linhas[1]) if len(linhas) >= 3 else None
            if m:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
                brutos.append((h1 * 3600 + m1 * 60 + s1 + ms1 / 1000, h2 * 3600 + m2 * 60 + s2 + ms2 / 1000,
                               " ".join(linhas[2:])))
    else:
        for linha in conteudo.splitlines():
            m = re.match(r"\[(\d+):(\d+):(\d+)\]\s*(.*)", linha)
            if m:
                h, mi, s, texto = m.groups()
                brutos.append((int(h) * 3600 + int(mi) * 60 + int(s), None, texto))
    trechos, quem = [], []
    for n, (inicio, fim, texto) in enumerate(brutos):
        if fim is None:
            fim = brutos[n + 1][0] if n + 1 < len(brutos) else inicio + 3
        m = re.match(r"(Pessoa \d+): (.*)$", texto)
        quem.append(m.group(1) if m else "")
        trechos.append((inicio, max(fim, inicio), m.group(2) if m else texto))
    if not trechos:
        raise RuntimeError(f"Não encontrei trechos de transcrição em {origem}.")
    return trechos, quem


class AbaRevisar(Aba):
    """Parte de arquivos já prontos: dá nomes às pessoas e gera a revisão em PDF, sem transcrever de novo."""

    def __init__(self, mestre):
        super().__init__(mestre, "*.txt *.srt *.md", "Transcrição (.txt, .srt) ou revisão (.md)")
        self.audio = tk.StringVar()
        self.separar = tk.BooleanVar(value=False)
        self.pessoas = tk.IntVar(value=0)
        self.descricao = tk.StringVar(value="")

        pad = {"padx": 8, "pady": 5}
        ttk.Label(self, text="Áudio (opcional):").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.audio, state="readonly").grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(self, text="Procurar...", command=self.procurar_audio).grid(row=2, column=2, **pad)
        vozes = ttk.Frame(self)
        vozes.grid(row=3, column=0, columnspan=3, sticky="w", **pad)
        ttk.Checkbutton(vozes, text="Separar por quem fala (precisa do áudio)",
                        variable=self.separar).pack(side="left", padx=(0, 15))
        ttk.Label(vozes, text="Nº de pessoas (0 = automático):").pack(side="left")
        ttk.Spinbox(vozes, from_=0, to=10, width=4, textvariable=self.pessoas).pack(side="left", padx=5)
        ttk.Label(self, textvariable=self.descricao, foreground="#1d4ed8", wraplength=620,
                  justify="left").grid(row=4, column=0, columnspan=3, sticky="w", **pad)
        self.montar_rodape(5, "Gerar PDF")

    def nome_base(self):
        base = super().nome_base()
        return base[:-len(" - revisado")] if base.endswith(" - revisado") else base

    def pasta_saida(self):
        return self.pasta.get()  # os arquivos já estão numa pasta: grava ali mesmo

    def procurar_audio(self):
        caminho = filedialog.askopenfilename(title="Áudio", filetypes=[("Áudio", AUDIOS), ("Todos os arquivos", "*.*")])
        if caminho:
            self.audio.set(os.path.normpath(caminho))

    def definir_arquivo(self, caminho):
        super().definir_arquivo(caminho)
        base, pasta = self.nome_base(), self.pasta.get()
        wav = os.path.join(pasta, base + ".wav")
        if not self.audio.get() and os.path.exists(wav):
            self.audio.set(wav)
        if caminho.lower().endswith(".md"):
            self.descricao.set("Vou só converter essa revisão (.md) em PDF.")
        elif os.path.exists(os.path.join(pasta, base + " - revisado.md")):
            self.descricao.set("Já existe uma revisão (.md): se você não mudar nada, só gero o PDF dela. "
                               "Se der nomes às pessoas, a revisão é refeita com a IA.")
        else:
            self.descricao.set("Vou dar nome às pessoas (se a transcrição tiver 'Pessoa N'), fazer a revisão com IA e gerar o PDF.")

    def validar(self):
        if self.separar.get():
            if not os.path.exists(self.audio.get()):
                messagebox.showwarning("Transcritor", "Escolha o áudio para poder separar por quem fala.")
                return False
            return self.garantir_token()
        return True

    def executar(self):
        arquivo, pasta, base = self.arquivo.get(), self.pasta.get(), self.nome_base()
        caminho_base = os.path.join(pasta, base)
        md = caminho_base + " - revisado.md"
        os.makedirs(pasta, exist_ok=True)

        if arquivo.lower().endswith(".md"):
            self.atualizar("Gerando o PDF...", 50)
            with open(arquivo, encoding="utf-8") as f:
                self.salvar_revisao(f.read(), caminho_base)
            self.atualizar("Concluído: PDF gerado", 100)
            self.after(0, self.acoes.grid)
            return

        self.atualizar("Lendo a transcrição...", 5)
        trechos, quem = ler_trechos(pasta, base, arquivo)
        aviso = ""
        mudou = False
        if self.separar.get():
            try:
                self.atualizar("Identificando quem fala (pode demorar alguns minutos)...", 20)
                temporario = os.path.join(tempfile.gettempdir(), "transcritor_preparado.wav")
                extrair_audio(self.audio.get(), temporario, False)
                falas = diarizar(temporario, self.pessoas.get(), os.environ["HF_TOKEN"])
                quem = [pessoa_do_trecho(i, f, falas) for i, f, _ in trechos]
                mudou = True
                os.remove(temporario)
            except Exception:
                log.exception("Falha ao separar por quem fala")
                aviso = " (sem separar vozes: veja o log)"
        if any(re.fullmatch(r"Pessoa \d+", p) for p in quem):
            quem, deu_nomes = self.nomear_pessoas(trechos, quem)
            mudou = mudou or deu_nomes

        def com_pessoa(n, t):
            return f"{quem[n]}: {t}" if quem[n] else t

        texto = "\n".join(f"[{formatar_tempo(i)}] {com_pessoa(n, t)}" for n, (i, _, t) in enumerate(trechos))
        if mudou:
            with open(caminho_base + ".txt", "w", encoding="utf-8") as f:
                f.write(texto)
            with open(caminho_base + ".srt", "w", encoding="utf-8") as f:
                for n, (i, fim, t) in enumerate(trechos):
                    f.write(f"{n + 1}\n{formatar_tempo_srt(i)} --> {formatar_tempo_srt(fim)}\n{com_pessoa(n, t)}\n\n")

        if not mudou and os.path.exists(md):
            self.atualizar("Convertendo a revisão que já existia em PDF...", 90)
            with open(md, encoding="utf-8") as f:
                self.salvar_revisao(f.read(), caminho_base)
            aviso += " · usei a revisão (.md) que já existia"
        else:
            aviso += self.revisar_e_gerar_pdf(caminho_base, texto)
        self.atualizar("Concluído: 100%" + aviso, 100)
        self.after(0, self.acoes.grid)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transcritor")
        try:
            # AppUserModelID próprio: a barra de tarefas usa o ícone do app, não o do pythonw
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("transcritor.app")
            self.iconbitmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcritor.ico"))
        except Exception:
            pass
        self.geometry("700x420")
        self.resizable(False, False)
        self.abas = ttk.Notebook(self)
        self.abas.pack(fill="both", expand=True, padx=8, pady=8)
        self.aba_transcrever = AbaTranscrever(self.abas)
        self.aba_extrair = AbaExtrair(self.abas, self.ir_para_transcricao)
        self.abas.add(self.aba_extrair, text="1. Extrair áudio do vídeo")
        self.abas.add(self.aba_transcrever, text="2. Transcrever áudio")
        self.abas.add(AbaRevisar(self.abas), text="3. Revisão e PDF")

    def ir_para_transcricao(self, caminho):
        self.aba_transcrever.definir_arquivo(caminho)
        self.abas.select(self.aba_transcrever)


if __name__ == "__main__":
    App().mainloop()
