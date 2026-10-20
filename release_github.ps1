# 创建 GitHub Release v1.0 并上传发布包
# 用法：powershell -ExecutionPolicy Bypass -File "F:\DS Harness\Color-Lab4.0\release_github.ps1"

$ErrorActionPreference = 'Stop'
$Owner = "yingqizhong02-jpg"
$Repo  = "Latis"
$ZIP   = "F:\DS Harness\Latis-1.0.zip"
$EXE   = "F:\DS Harness\Latis-1.0\Latis.exe"

Write-Host "=== 1. 取 token（不打印内容）==="
$env:GIT_TERMINAL_PROMPT = "0"
$cred = "protocol=https`nhost=github.com`n`n" | & git credential fill 2>$null
$token = ""
foreach ($l in $cred) { if ($l -like "password=*") { $token = $l.Substring(9) } }
if (-not $token) { throw "凭据管理器里没有 github token" }
Write-Host "  OK  token 长度 $($token.Length)"
$H = @{ Authorization = "Bearer $token"; Accept = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28" }

Write-Host "`n=== 2. 本地发布物 ==="
if (-not (Test-Path $EXE)) { throw "找不到 $EXE" }
if (-not (Test-Path $ZIP)) { throw "找不到 $ZIP" }
$exeHash = (Get-FileHash $EXE -Algorithm SHA256).Hash
$zipHash = (Get-FileHash $ZIP -Algorithm SHA256).Hash
Write-Host ("  Latis.exe  {0:N1} MB  {1}" -f ((Get-Item $EXE).Length/1MB), $exeHash)
Write-Host ("  zip        {0:N1} MB  {1}" -f ((Get-Item $ZIP).Length/1MB), $zipHash)

Write-Host "`n=== 3. 仓库确认 ==="
$me = Invoke-RestMethod -Uri "https://api.github.com/user" -Headers $H -Method Get
Write-Host "  登录: $($me.login)"
$branch = Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Repo/branches/main" -Headers $H -Method Get
Write-Host "  main: $($branch.commit.sha.Substring(0,7))  $($branch.commit.commit.message.Split("`n")[0])"

Write-Host "`n=== 4. 建 Release v1.0 ==="
$body = @"
RAW 照片 → LUT → 带 ICC 的成品图。免安装。

## 下载

下载 ``Latis-1.0.zip``，解压到任意位置，双击 ``Latis.exe``。
不需要装 Python，不需要管理员权限。配置和 LUT 都存在 exe 旁边，删掉文件夹就是卸载。

首次运行可能被 SmartScreen 拦（未做代码签名）：右键 exe → 属性 → 勾「解除锁定」→ 确定。

## 特性

- **原厂 RAW 直接进** —— 基于 rawpy（LibRaw）解码，覆盖主流相机厂商的 RAW 与专有压缩格式
- **全程 32 位** —— 解码后在 float32 线性域完成全部运算，多级 CST 串联不累积量化误差
- **LUT 不挑输入空间** —— 每份 LUT 只声明输入空间，内置 CST 负责把源转过去；不同输入空间的 LUT 可混用
- **色彩空间全程可见** —— 源 → LUT 输入 → LUT 输出 → 文件目标，四段都写在界面上
- **导出文件自洽** —— 文件里的色彩空间标记和像素值对得上

## 校验

Latis.exe SHA256:
Latis-1.0.zip SHA256:

完整逐文件哈希见压缩包内 SHA256SUMS.txt（1798 条）。
发布时已用 Windows Defender 完整扫描，未发现威胁。

## 系统要求

Windows 10/11 x64。不需要 Python。

## 许可

MIT。使用前请读压缩包内「免责声明.txt」。
"@
$body = $body.Replace("Latis.exe SHA256:", "Latis.exe SHA256:")
$body += ""
# 把两个哈希补进正文
$body = $body -replace "Latis\.exe SHA256:`n", "Latis.exe SHA256:`n``````text`n$exeHash`n```````n"
$body = $body -replace "Latis-1\.0\.zip SHA256:`n", "Latis-1.0.zip SHA256:`n``````text`n$zipHash`n```````n"

$payload = @{
    tag_name = "v1.0"; target_commitish = "main"; name = "Latis 1.0"
    body = $body; draft = $false; prerelease = $false
} | ConvertTo-Json -Depth 5
$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
$rel = Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Repo/releases" `
       -Headers $H -Method Post -Body $bytes -ContentType "application/json; charset=utf-8"
Write-Host "  OK  $($rel.html_url)"

Write-Host "`n=== 5. 上传 zip（51 MB，请稍候）==="
$upUrl = $rel.upload_url.Split('{')[0] + "?name=Latis-1.0.zip"
$asset = Invoke-RestMethod -Uri $upUrl -Headers $H -Method Post -InFile $ZIP `
         -ContentType "application/zip" -TimeoutSec 1800
Write-Host ("  OK  {0}  {1:N1} MB" -f $asset.name, ($asset.size/1MB))
Write-Host "  $($asset.browser_download_url)"
if ($asset.digest) { Write-Host "  digest: $($asset.digest)" }

Write-Host "`n=== 6. 最终确认 ==="
$final = Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Repo/releases/tags/v1.0" -Headers $H -Method Get
Write-Host "  标签: $($final.tag_name)"
Write-Host "  标题: $($final.name)"
Write-Host "  状态: $(if ($final.draft) { '草稿（未发布）' } else { '已发布' })"
Write-Host "  资产: $(($final.assets | ForEach-Object { $_.name }) -join ', ')"
Write-Host "  页面: $($final.html_url)"
