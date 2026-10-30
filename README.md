<p align="center">
  <strong>🌐 Idioma / Language:</strong>
  🇧🇷 <strong>Português</strong> | <a href="README.en.md">🇺🇸 English</a>
</p>

# Transcritor

Transcrição de áudio e vídeo em português com o [Whisper](https://github.com/SYSTRAN/faster-whisper), rodando **localmente** na sua placa de vídeo. O programa extrai o áudio de um vídeo, transcreve, separa quem fala e gera um resumo em PDF com IA, tudo por uma interface simples para Windows.

## O que ele faz

O programa tem três abas:

1. **Extrair áudio do vídeo**: converte um vídeo (`.mp4`, `.mkv`, `.mov`, `.webm`...) em um `.wav` de 16 kHz mono, o formato nativo do Whisper. Opcionalmente normaliza o volume e corta graves.
2. **Transcrever áudio**: transcreve com o Whisper (modelos *small*, *medium* ou *large-v3*) e gera:
   - `.txt` com o tempo de cada trecho;
   - `.srt` (legenda com início e fim);
   - **separação por quem fala** (opcional): identifica "Pessoa 1", "Pessoa 2"... e abre uma janela para você dar o nome de cada um;
   - **revisão com IA** (opcional): um PDF com título, resumo por assunto, tarefas, texto organizado em ordem e a lista de trechos incertos.
3. **Revisão e PDF**: parte de arquivos já prontos (`.txt`, `.srt` ou a revisão `.md`), sem transcrever de novo. Dá nomes às pessoas, refaz a revisão e gera o PDF.

Tudo é gravado numa pasta com o nome do arquivo original:

```
reuniao.mkv
reuniao\
    reuniao.wav
    reuniao.txt
    reuniao.srt
    reuniao - revisado.pdf
```

## Requisitos

| | |
|---|---|
| Sistema | Windows 10/11 (usa o Edge ou o Chrome para gerar o PDF) |
| Python | **3.10 a 3.13** (testado com a 3.13) |
| Placa de vídeo | NVIDIA, de preferência (testado numa RTX 2060 de 6 GB). Sem ela, roda no processador, bem mais devagar |
| Espaço em disco | Cerca de 10 GB livres (PyTorch com CUDA + modelos) |

O restante (ffmpeg, PyTorch, Whisper, pyannote) o script de instalação cuida.

## Instalação

```powershell
git clone https://github.com/gustavocomin/Transcritor.git
cd Transcritor
powershell -ExecutionPolicy Bypass -File .\instalar.ps1
```

O script:
1. confere o Python;
2. instala o **ffmpeg** pelo `winget`, se necessário;
3. cria o ambiente virtual `.venv` na pasta do projeto;
4. instala o **PyTorch** (com CUDA, se houver placa NVIDIA) e depois as dependências do `requirements.txt`;
5. cria o atalho **Transcritor** no Menu Iniciar.

Opções: `-Cpu` instala o PyTorch só de CPU, e `-SemAtalho` não cria o atalho.

Para abrir: pelo Menu Iniciar (digite "Transcritor") ou com duplo clique em `Transcritor.bat`.

Na primeira vez que você usar cada modelo, ele é baixado automaticamente (o *large-v3* tem cerca de 3 GB). Os modelos ficam no cache do Hugging Face do seu usuário (`%USERPROFILE%\.cache\huggingface`).

**Atualizar:** `git pull` e rode `instalar.ps1` de novo (ele reaproveita o que já está instalado).

**Desinstalar:** apague a pasta do projeto e o atalho `Transcritor` do Menu Iniciar. Se quiser, apague também a variável de ambiente `HF_TOKEN` e os modelos em `%USERPROFILE%\.cache\huggingface`.

## Qual modelo escolher

| Opção na tela | Modelo | Quando usar |
|---|---|---|
| Rápido | `small` | Testes e áudios limpos; é o mais leve |
| Médio | `medium` | Meio-termo entre velocidade e qualidade |
| Alta qualidade | `large-v3` | Áudio ruim, vozes distantes ou várias pessoas; é o padrão |

O *large-v3* cabe numa placa de 6 GB em `float16`. Se faltar memória da placa, o programa tenta `int8_float16` e, em último caso, o processador. Se a placa falhar no meio da transcrição, ele refaz tudo no processador e avisa na tela.

## Separar por quem fala (Hugging Face)

A separação usa o modelo [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1), que exige uma conta gratuita:

1. Crie a conta em <https://huggingface.co/join>.
2. Abra a página do modelo, preencha o formulário (os campos "Company/university" e "Website" aceitam, por exemplo, "Independent" e o seu perfil) e clique em **Agree and access repository**.
3. Gere um token em <https://huggingface.co/settings/tokens> (tipo **Read**).
4. No programa, marque **Separar por quem fala**. Na primeira vez ele pede o token e o guarda na variável de ambiente `HF_TOKEN` do seu usuário. Depois não pergunta mais.

Informe o número de pessoas quando souber (ajuda o modelo); `0` deixa automático. Falas sobrepostas, vozes parecidas e quem está longe do microfone são os casos em que ele mais erra, então trate os rótulos como aproximados.

## Revisão com IA

A revisão chama, nesta ordem:
1. o **Claude Code** (`claude -p`);
2. se falhar (limite de uso, erro), o **Codex** (`codex exec`);
3. se os dois falharem, o programa avisa na tela, e a transcrição (`.txt` e `.srt`) já está salva. Você pode gerar a revisão depois pela aba 3.

Para isso, o Claude Code e/ou o Codex precisam estar instalados e com login feito. Sem eles, desmarque "Gerar revisão com IA", e o resto funciona normalmente.

**Privacidade:** o texto da transcrição é enviado ao serviço da IA usada na revisão. Desmarque a opção se o conteúdo for sensível.

O pedido enviado à IA está em [`prompt_revisao.txt`](prompt_revisao.txt) e pode ser editado à vontade (por exemplo, para pedir outro formato de resumo).

## Problemas comuns

- **O log:** erros e etapas ficam em `transcritor.log`, na pasta do programa.
- **`Library cublas64_12.dll is not found`:** o PyTorch com CUDA não foi instalado. Rode `instalar.ps1` de novo e confira se há uma placa NVIDIA e o driver atualizado.
- **O PDF não sai:** é preciso ter o Edge ou o Chrome. Se falhar, a revisão é salva em `.md`.
- **A separação de vozes dá erro 401:** falta aceitar os termos do modelo `community-1` com a mesma conta do token.

## Estrutura

| Arquivo | Função |
|---|---|
| `transcritor.py` | O programa (interface, transcrição, vozes, revisão e PDF) |
| `prompt_revisao.txt` | Instruções enviadas à IA para a revisão |
| `instalar.ps1` | Instalação das dependências e do atalho |
| `requirements.txt` | Dependências Python (o PyTorch é instalado antes, pelo script) |
| `Transcritor.bat` | Abre o programa sem instalar o atalho |
| `LICENSE` | Licença MIT |

## Licença

Distribuído sob a licença [MIT](LICENSE). O programa não inclui nem redistribui modelos nem o ffmpeg: cada pessoa os obtém na instalação, e os modelos têm as próprias licenças. O modelo de separação de vozes `pyannote/speaker-diarization-community-1` usa a licença **CC-BY-4.0** e exige atribuição aos autores do [pyannote](https://github.com/pyannote/pyannote-audio).
