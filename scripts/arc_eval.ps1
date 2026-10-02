# Прогон на машине с видеокартой Intel (Arc) под Windows: декодер синтеза
# на OpenVINO против DirectML и процессора (experiments/27), а с -Full —
# ещё и сквозной синтез выпущенным voicy.exe на Vulkan + DirectML.
#
#   powershell -ExecutionPolicy Bypass -File scripts\arc_eval.ps1          # ~1 ГБ загрузки
#   powershell -ExecutionPolicy Bypass -File scripts\arc_eval.ps1 -Full    # + бинарник и модели, ~8 ГБ
#
# Нужны только интернет и драйвер видеокарты; uv и Python скрипт ставит сам.
# Итог — arc-eval\arc-eval.zip: его и привезти.
param([switch]$Full, [string]$Out = "arc-eval")

# не Stop: Windows PowerShell считает ошибкой любую строку в stderr программы,
# а ONNX Runtime и uv пишут туда предупреждения и ход работы
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$Res = Join-Path $Out "results"
New-Item -ItemType Directory -Force $Res | Out-Null
Start-Transcript -Path (Join-Path $Res "transcript.txt") -Append | Out-Null

function Note($s) { Write-Host "`n== $s" -ForegroundColor Cyan }

Note "машина"
$sys = @(
  Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate, AdapterRAM | Format-List | Out-String
  Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List | Out-String
  (Get-CimInstance Win32_OperatingSystem).Caption + " " + (Get-CimInstance Win32_OperatingSystem).Version
  "RAM, ГБ: " + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
)
$sys | Tee-Object (Join-Path $Res "system.txt")

Note "uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
  $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
}
uv --version

Note "модели декодера (~440 МБ)"
$Models = Join-Path $Out "models"
New-Item -ItemType Directory -Force $Models | Out-Null
$hf = "https://huggingface.co/sknyazev/qwen3-tts-12hz-1.7b-ru-stress-gguf/resolve/main"
foreach ($f in "qwen3_tts_decoder.fp16.onnx", "qwen3_tts_codec_encoder.fp32.onnx", "qwen3_tts_codec_encoder.fp32.onnx.data") {
  $p = Join-Path $Models $f
  if (-not (Test-Path $p)) { curl.exe -L --fail -o $p "$hf/$f"; if ($LASTEXITCODE) { throw "не скачался $f" } }
}

Note "окружения Python"
$envs = [ordered]@{
  "cpu" = @("onnxruntime==1.24.1")
  "ov"  = @("onnxruntime-openvino==1.24.1", "openvino==2025.4.1")
  "dml" = @("onnxruntime-directml==1.24.4")
}
foreach ($e in $envs.Keys) {
  $v = Join-Path $Out "venv-$e"
  # без -q: первый раз uv качает Python и пакеты, и молчание выглядит как зависание
  if (-not (Test-Path "$v\Scripts\python.exe")) {
    # недоделанное окружение после Ctrl+C: uv спросил бы, заменять ли его, и ждал ответа
    if (Test-Path $v) { Remove-Item -Recurse -Force $v }
    Write-Host "окружение ${e}: Python 3.12"
    uv venv -p 3.12 $v
  }
  Write-Host "окружение ${e}: пакеты $($envs[$e] -join ', ')"
  uv pip install --python "$v\Scripts\python.exe" numpy @($envs[$e])
  if ($LASTEXITCODE) { throw "не встало окружение $e" }
}

Note "устройства OpenVINO"
& "$Out\venv-ov\Scripts\python.exe" -c "import openvino as ov; c = ov.Core(); [print(d, '-', c.get_property(d, 'FULL_DEVICE_NAME')) for d in c.available_devices]" 2>&1 |
  Tee-Object (Join-Path $Res "openvino-devices.txt")

Note "декодер: процессор, OpenVINO (CPU и GPU), DirectML"
$eval = Join-Path $Root "scripts\decoder_ep_eval.py"
$wav = Join-Path $Root "server\voices\turgenev.wav"
$json = Join-Path $Res "decoder.jsonl"
$runs = @(
  @("cpu", "cpu", @()),
  @("ov", "ov-cpu", @()),
  @("ov", "ov-gpu", @()),
  @("ov", "ov-gpu", @("--precision", "FP32")),
  @("dml", "dml", @())
)
foreach ($r in $runs) {
  $py = "$Out\venv-$($r[0])\Scripts\python.exe"
  $tag = $r[1] + ($(if ($r[2].Count) { "-" + $r[2][1].ToLower() } else { "" }))
  # запись: скорость и звук на настоящей речи; случайные коды: 16 с, длиннее
  & $py $eval $r[1] @($r[2]) --models $Models --wav $wav --out (Join-Path $Res "$tag.wav") --json $json 2>&1 |
    Where-Object { $_ -notmatch "W:onnxruntime" } | Tee-Object -Append (Join-Path $Res "decoder.txt")
  & $py $eval $r[1] @($r[2]) --models $Models --json $json 2>&1 |
    Where-Object { $_ -notmatch "W:onnxruntime" } | Tee-Object -Append (Join-Path $Res "decoder.txt")
}

if ($Full) {
  Note "сквозной синтез: voicy.exe, Vulkan + DirectML"
  $exe = Join-Path $Out "voicy.exe"
  if (-not (Test-Path $exe)) {
    curl.exe -L --fail -o $exe https://github.com/olluorg/voicy/releases/latest/download/voicy-windows-x64.exe
  }
  $env:VOICY_DEVICE = "vulkan"
  & $exe setup --yes 2>&1 | Tee-Object (Join-Path $Res "setup.txt")
  & $exe up 2>&1 | Tee-Object (Join-Path $Res "up.txt")
  & $exe status 2>&1 | Tee-Object (Join-Path $Res "status.txt")
  $text = Join-Path $Root "scripts\arc_eval.txt"
  foreach ($i in 0..3) {
    & $exe say "@$text" (Join-Path $Res "say-$i.opus") 2>&1 | Tee-Object -Append (Join-Path $Res "say.txt")
  }
  & $exe hear (Join-Path $Res "say-3.opus") 2>&1 | Tee-Object (Join-Path $Res "hear.txt")
  & $exe status 2>&1 | Tee-Object -Append (Join-Path $Res "status.txt")
  & $exe down 2>&1 | Out-Null
}

Stop-Transcript | Out-Null
$zip = Join-Path $Out "arc-eval.zip"
Compress-Archive -Force -Path "$Res\*" -DestinationPath $zip
Note "готово: $zip"
