# Latis 发布到 GitHub —— 一键脚本
#
# 用法：右键 →「使用 PowerShell 运行」，或在终端里：
#     powershell -ExecutionPolicy Bypass -File "F:\DS Harness\Color-Lab4.0\publish.ps1"
#
# 它会：设置 git 身份 → 询问仓库地址 → 推送 → 打开 Release 页面
# 推送时会由 Git Credential Manager 弹出浏览器，登录一次即可。

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$ProgressPreference = "SilentlyContinue"

function Line($t) { Write-Host "`n$t" -ForegroundColor Cyan }
function Ok($t)   { Write-Host "  OK  $t" -ForegroundColor Green }
function Warn($t) { Write-Host "  !!  $t" -ForegroundColor Yellow }

Line "=== 1. 检查前置 ==="
if (-not (Test-Path ".git")) { throw "当前目录不是 git 仓库：$PSScriptRoot" }
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
$commit = (git log --oneline -1)
$files = (git ls-files | Measure-Object).Count
Ok "分支 $branch  提交 $commit  文件 $files 个"

Line "=== 2. git 身份 ==="
$name  = (git config --global user.name).Trim()
$email = (git config --global user.email).Trim()
if (-not $name)  { $name  = Read-Host "  你的名字（会写进提交记录）" }
if (-not $email) { $email = Read-Host "  你的邮箱（建议与 GitHub 一致）" }
git config --global user.name $name
git config --global user.email $email
Ok "$name <$email>"

Line "=== 3. 仓库地址 ==="
$existing = (git remote get-url origin 2>$null)
if ($existing) {
    $url = $existing
    Ok "已有 remote: $url"
} else {
    Write-Host "  先在 GitHub 网页建一个**空**仓库（不要勾 README / gitignore / license）"
    Write-Host "  然后粘贴它的 HTTPS 地址，例如："
    Write-Host "      https://github.com/你的用户名/latis.git"
    $url = (Read-Host "  仓库地址").Trim()
    if (-not $url) { throw "没有仓库地址，退出" }
    git remote add origin $url
    Ok "已添加 origin"
}

Line "=== 4. 推送 ==="
Write-Host "  接下来会弹出浏览器让你登录 GitHub（Git Credential Manager）"
Write-Host "  登录后窗口会自动关闭，推送继续"
git push -u origin $branch
if ($LASTEXITCODE -ne 0) { throw "推送失败，见上方 git 输出" }
Ok "已推送"

$web = $url -replace '\.git$', ''
Ok "仓库地址: $web"

Line "=== 5. 下一步：发 Release ==="
$releaseUrl = "$web/releases/new?tag=v1.0&title=Latis+1.0"
Write-Host "  1) 浏览器会打开新建 Release 页面"
Write-Host "  2) 上传这个文件："
Write-Host "         F:\DS Harness\Latis-1.0.zip   (51.4 MB)"
Write-Host "  3) 描述可以贴 README 里的「特性」一节"
Write-Host "  4) 点 Publish release"
Start-Process $releaseUrl

Line "=== 完成 ==="
Write-Host "  源代码: $web"
Write-Host "  发布包: $web/releases"
