# Instala o Transcritor: ffmpeg, ambiente virtual (.venv), PyTorch, dependências e atalho no Menu Iniciar.
# Pré-requisito: Python 3.10 a 3.13 instalado (testado com a 3.13).
# Uso: powershell -ExecutionPolicy Bypass -File .\instalar.ps1
#   -SemAtalho  não cria o atalho no Menu Iniciar
#   -Cpu        instala o PyTorch só de CPU (sem placa NVIDIA)
param(
    [switch]$SemAtalho,
    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'
$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $raiz

# Versões testadas do PyTorch (CUDA 12.6)
$torch = 'torch==2.14.1'
$torchaudio = 'torchaudio==2.11.0'
$indiceCuda = 'https://download.pytorch.org/whl/cu126'

function Passo($texto) { Write-Host "`n==> $texto" -ForegroundColor Cyan }
function Falha($texto) { Write-Host "`nERRO: $texto" -ForegroundColor Red; exit 1 }

function Atualizar-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

function Executar($arquivo, [string[]]$argumentos) {
    # O Windows PowerShell 5.1 trata qualquer texto em stderr (avisos do pip) como erro com 'Stop'
    $antes = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $arquivo @argumentos
    $codigo = $LASTEXITCODE
    $ErrorActionPreference = $antes
    if ($codigo -ne 0) { Falha "O comando falhou: $arquivo $($argumentos -join ' ')" }
}

# 1. Python
Passo 'Verificando o Python'
$pythonExe = $null
$pythonArgs = @()
foreach ($candidato in @('python', 'py')) {
    $comando = Get-Command $candidato -ErrorAction SilentlyContinue
    if (-not $comando) { continue }
    $argumentos = if ($candidato -eq 'py') { @('-3') } else { @() }
    $antes = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $saida = & $comando.Source @argumentos -c 'import sys; print(sys.version_info[0], sys.version_info[1])' 2>$null
    $codigo = $LASTEXITCODE
    $ErrorActionPreference = $antes
    if ($codigo -eq 0 -and $saida) {
        $versao = ($saida -split ' ')[0] + '.' + ($saida -split ' ')[1]
        $pythonExe = $comando.Source
        $pythonArgs = $argumentos
        break
    }
}
if (-not $pythonExe) { Falha 'Python não encontrado. Instale a versão 3.13 em https://www.python.org/downloads/ (marque "Add python.exe to PATH") e rode o script de novo.' }
$partes = $versao.Split('.')
if ([int]$partes[0] -ne 3 -or [int]$partes[1] -lt 10 -or [int]$partes[1] -gt 13) {
    Falha "Python $versao não é suportado. Use a versão 3.10 a 3.13 (testado com a 3.13)."
}
if ($versao -ne '3.13') { Write-Host "Aviso: este programa foi testado só com o Python 3.13; você está com o $versao." -ForegroundColor Yellow }
Write-Host "Python $versao encontrado."

# 2. ffmpeg
Passo 'Verificando o ffmpeg'
if (-not ((Get-Command ffmpeg -ErrorAction SilentlyContinue) -and (Get-Command ffprobe -ErrorAction SilentlyContinue))) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Falha 'ffmpeg não encontrado e o winget não está disponível. Instale o ffmpeg (https://ffmpeg.org/download.html), coloque-o no PATH e rode o script de novo.'
    }
    Write-Host 'Instalando o ffmpeg pelo winget...'
    Executar 'winget' @('install', '--id', 'Gyan.FFmpeg.Essentials', '-e', '--accept-source-agreements', '--accept-package-agreements')
    Atualizar-Path
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Falha 'ffmpeg instalado, mas ainda não está no PATH desta janela. Feche e abra o terminal e rode o script de novo.'
    }
}
Write-Host 'ffmpeg encontrado.'

# 3. Ambiente virtual
Passo 'Criando o ambiente virtual (.venv)'
$venvPython = Join-Path $raiz '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Executar $pythonExe (@($pythonArgs) + @('-m', 'venv', '.venv'))
}
Executar $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip')

# 4. PyTorch (antes do resto, para o pip não trazer uma versão só de CPU)
Passo 'Instalando o PyTorch'
$temNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($Cpu -or -not $temNvidia) {
    Write-Host 'Sem placa NVIDIA detectada (ou -Cpu): instalando o PyTorch de CPU. A transcrição será bem mais lenta.' -ForegroundColor Yellow
    Executar $venvPython @('-m', 'pip', 'install', $torch, $torchaudio)
} else {
    Write-Host 'Placa NVIDIA detectada: instalando o PyTorch com CUDA (download grande, alguns GB).'
    Executar $venvPython @('-m', 'pip', 'install', $torch, $torchaudio, '--index-url', $indiceCuda)
}

# 5. Dependências
Passo 'Instalando as dependências (requirements.txt)'
Executar $venvPython @('-m', 'pip', 'install', '-r', 'requirements.txt')

# 6. Verificação
Passo 'Verificando a instalação'
Executar $venvPython @('-c', 'import faster_whisper, pyannote.audio, markdown_it, torch; print(''PyTorch'', torch.__version__, ''| CUDA disponivel:'', torch.cuda.is_available())')

# 7. Atalho no Menu Iniciar
if (-not $SemAtalho) {
    Passo 'Criando o atalho no Menu Iniciar'
    $atalho = Join-Path ([Environment]::GetFolderPath('Programs')) 'Transcritor.lnk'
    $shell = (New-Object -ComObject WScript.Shell).CreateShortcut($atalho)
    $shell.TargetPath = Join-Path $raiz '.venv\Scripts\pythonw.exe'
    $shell.Arguments = '"' + (Join-Path $raiz 'transcritor.py') + '"'
    $shell.WorkingDirectory = $raiz
    $shell.IconLocation = Join-Path $raiz 'transcritor.ico'
    $shell.Description = 'Transcreve áudio e vídeo com Whisper'
    $shell.Save()
    Write-Host "Atalho criado: $atalho"
}

Write-Host "`nPronto! Abra o 'Transcritor' pelo Menu Iniciar (ou dê duplo clique em Transcritor.bat)." -ForegroundColor Green
Write-Host 'Para separar quem fala, veja a seção do Hugging Face no README.'
