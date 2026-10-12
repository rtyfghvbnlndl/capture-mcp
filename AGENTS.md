# capture-mcp：给 agent 的安装与接入说明

本包是一个 MCP server（stdio）。它从 USB HDMI 采集卡读取画面，采集卡在系统里被识别成摄像头，接的是另一台电脑的 HDMI 输出。多模态模型通过它查看那台电脑的屏幕。只读，不能控制键鼠。

按下面的步骤做，每一步都有验证方法。完整说明见 `README.md`。

## 1. 安装

前提：Python 3.10 或更高版本。

**方式 A：解压后运行安装脚本（推荐）**

```powershell
tar -xzf capture_mcp-0.1.0.tar.gz
cd capture_mcp-0.1.0
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

```sh
tar -xzf capture_mcp-0.1.0.tar.gz && cd capture_mcp-0.1.0 && sh ./install.sh
```

- 安装位置：venv 默认建在 `~/.capture-mcp/venv`。
- 脚本做的事：安装依赖，检查导入，列出视频设备，最后把客户端配置写进同目录的 `mcp-config.json`。
- 需要 OCR 工具（`read_text`）的话，加 `-Ocr`（Windows）或 `--ocr`（Linux/macOS）。

**方式 A2：从 GitHub 克隆后运行安装脚本**

```sh
git clone https://github.com/rtyfghvbnlndl/capture-mcp && cd capture-mcp
```

然后同方式 A 运行 `install.ps1` 或 `install.sh`。

**方式 C：用 pip 直接从 GitHub 安装**

```sh
python -m venv ~/.capture-mcp/venv
~/.capture-mcp/venv/bin/python -m pip install "capture-mcp[enum] @ git+https://github.com/rtyfghvbnlndl/capture-mcp"
```

**方式 B：不解压，用 pip 直接安装压缩包**

```sh
python -m venv ~/.capture-mcp/venv
~/.capture-mcp/venv/bin/python -m pip install "capture_mcp-0.1.0.tar.gz[enum]"
```

Windows 上把 `bin/python` 换成 `Scripts\python.exe`。

**Windows 注意**：venv 路径要短。路径太深时，OpenCV 会报 `DLL load failed ... 文件名或扩展名太长`。

**验证安装**：运行 `<venv python> -m capture_mcp --list`，应列出视频设备，并说明会用哪个设备、为什么。

## 2. 接入 MCP 客户端

客户端配置用 `mcpServers` 格式，`command` 填 **venv 里 python 的绝对路径**：

```json
{
  "mcpServers": {
    "capture-card": {
      "command": "C:\\Users\\<user>\\.capture-mcp\\venv\\Scripts\\python.exe",
      "args": ["-m", "capture_mcp"],
      "env": {}
    }
  }
}
```

- `env` 留空即可：设备由模型通过 `list_devices` 和 `select_device` 自己找，选过的设备会被记住。
- 只有想强制固定设备时，才设置 `CAPTURE_DEVICE`（序号或名称片段）。
- 其他环境变量见 `README.md`。

各客户端在哪里填这份配置：

| 客户端 | 位置 |
|---|---|
| Claude Code | `claude mcp add capture-card -- <venv python> -m capture_mcp` |
| LM Studio | Program → Install → Edit `mcp.json` |
| Cherry Studio | 设置 → MCP 服务器 → 添加，类型选 stdio |
| 其他 | 填进该客户端的 `mcpServers` 配置 |

**验证接入**：客户端里应该出现这些工具：`screenshot`、`zoom`、`wait_for_change`、`list_devices`、`select_device`。装了 OCR 的话还有 `read_text`。

## 3. 让模型用得好

1. 把 `SYSTEM_PROMPT.md` 里的提示词放进客户端的 system prompt。
2. 模型必须同时支持**图片输入**和**工具调用**。
3. 客户端如果有"只保留最近 N 张图片"这类设置，建议打开，能避免截图把上下文撑爆。

## 4. 首次使用检查清单

1. 采集卡插到本机，另一台电脑的 HDMI 线插进采集卡。
2. 另一台电脑：按 `Win+P` 选"复制"，分辨率设为 16:9（如 1920×1080），关闭睡眠。
3. 调用 `list_devices`。采集卡应显示为 `capture card (likely) | picture OK`。
   - 看不出哪个是采集卡时，调用 `list_devices(preview=true)` 看缩略图。
4. 如果当前用的不是采集卡，调用 `select_device(device="<序号>")`。这个选择会被记住。
5. 调用 `screenshot`，确认图里是另一台电脑的屏幕。

## 常见错误

| 现象 | 处理 |
|---|---|
| `Cannot open video device` / `delivers no frames` | 设备被 OBS、相机应用或浏览器占用了。关掉这些程序再试 |
| `No capture card recognized` | 采集卡没插好，或者驱动没把它识别成采集卡。用 `list_devices(preview=true)` 看画面，再用 `select_device` 手动选 |
| `NO IMAGE: screen is completely black` | 另一台电脑没信号、在睡眠，或 HDMI 没插好 |
| 1080p 很卡 | 看 `list_devices` 里的编码，应为 MJPG；如果不是，设置 `CAPTURE_BACKEND=msmf` 或降低分辨率 |
| 模型说看不到图 | 客户端可能没把工具返回的图片交给模型。设置 `CAPTURE_SAVE_DIR` 把图存下来排查 |
| 安装时 pip 报 `SSLEOFError` / `UNEXPECTED_EOF_WHILE_READING` | 本机代理（`HTTPS_PROXY`）和 Python 的 TLS 不兼容。安装时临时清空代理：PowerShell 里先执行 `$env:HTTPS_PROXY=""; $env:HTTP_PROXY=""` 再运行安装脚本 |
| 摄像头/采集卡打开了但始终全黑 | Windows 设置 → 隐私和安全性 → 摄像头 → 允许桌面应用访问摄像头；笔记本摄像头还要检查物理遮挡片 |
| Linux 上没有设备 | 当前用户需要有 `/dev/video*` 的访问权限，一般加入 `video` 组即可 |
