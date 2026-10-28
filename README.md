# capture-mcp

把 USB HDMI 采集卡（系统里识别成摄像头）接到另一台电脑的 HDMI 输出，通过 MCP 让本地多模态模型"看"那台电脑的屏幕。只读，不控制键鼠。

专门针对 **4bit 量化的 ~27B 多模态小模型** 做了取舍：少量扁平工具、宽松参数解析、带标签网格代替像素坐标、无信号时只回文字不回图。

## 安装

直接从 GitHub 安装（不用下载源码）：

```bash
pip install "capture-mcp[enum] @ git+https://github.com/rtyfghvbnlndl/capture-mcp"
```

或者克隆后运行安装脚本：

```bash
git clone https://github.com/rtyfghvbnlndl/capture-mcp
```

安装脚本会在 `~/.capture-mcp/venv` 建 venv、安装、列出设备，并生成 `mcp-config.json`。
给 agent 看的分步说明见 [AGENTS.md](AGENTS.md)，系统提示词见 [SYSTEM_PROMPT.md](SYSTEM_PROMPT.md)。

```bash
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Linux / macOS：`sh ./install.sh`。手动安装：

```bash
python -m venv C:\venvs\capmcp
C:\venvs\capmcp\Scripts\python -m pip install .[enum]
```

- `[enum]` 安装 `cv2-enumerate-cameras`，用于拿到设备名称（强烈建议，否则只能按索引猜）。
- 读屏（OCR）：`pip install .[ocr]`，装了才有 `read_screen` 工具。Windows 上用系统自带 OCR，其他系统用 RapidOCR；Windows 上也想用 RapidOCR 就装 `.[ocr-rapid]`。安装脚本加 `-Ocr`（或给不支持图片的模型用 `-TextOnly`）。
- **Windows 注意**：venv 路径不要太深。路径过长时 OpenCV 会报 `DLL load failed: 文件名或扩展名太长`。

## 自检（不经过 MCP）

```bash
C:\venvs\capmcp\Scripts\python -m capture_mcp --list
```

```bash
C:\venvs\capmcp\Scripts\python -m capture_mcp --probe
```

`--list` 会显示设备清单、记住的设备，以及会用哪个设备、为什么。`--probe` 会打印选中的设备、实际分辨率/FPS/编码、按需解码速度、信号状态，并保存 `probe.jpg` 和带网格的 `probe_grid.jpg`。
确认：选中的是采集卡而不是笔记本摄像头；画面是另一台电脑；拔掉 HDMI 后应报 "screen is completely black / single flat color"。

## 客户端配置

通用 `mcpServers` 格式（LM Studio `mcp.json`、Cherry Studio、Claude Desktop 等都是这个结构）：

```json
{
  "mcpServers": {
    "capture-card": {
      "command": "C:\\venvs\\capmcp\\Scripts\\python.exe",
      "args": ["-m", "capture_mcp"],
      "env": {}
    }
  }
}
```

不用填设备：让模型自己找（见下一节）。只有想**强制**固定设备时才在 `env` 里加 `"CAPTURE_DEVICE": "USB Video"`，它的优先级高于模型记住的选择。

调试可用 MCP Inspector：`npx @modelcontextprotocol/inspector C:\venvs\capmcp\Scripts\python.exe -m capture_mcp`

## 设备由 agent 自己找和设置

不需要手动配置设备。选择顺序：

1. 本次运行中模型用 `select_device` 选过的设备
2. 客户端配置里的 `CAPTURE_DEVICE`（如果填了）
3. 上次 `select_device` 记住的设备（存在 `~/.capture-mcp/settings.json`）。按设备路径和 VID/PID 匹配，换序号、两块同名 "USB Video" 都能认对
4. 自动：优先名字像采集卡的设备，跳过笔记本摄像头；有多块采集卡时选**有画面**的那块

自动选错或报错时，模型会调用 `list_devices` 看清单和信号，再调用 `select_device` 切换，之后会一直记住。
也可以直接对模型说："列出视频设备，帮我选出接着另一台电脑的那个"。

## 工具

| 工具 | 作用 | 参数（全部可选） |
|---|---|---|
| `screenshot` | 整屏截图，叠加 A1..H5 网格 | `detail`: low/normal/high；`grid`: true/false |
| `zoom` | 放大局部读小字（小区域最多放大 3 倍，不画网格） | `region`, `detail`（默认 high） |
| `wait_for_change` | 服务端轮询，画面/区域变化才返回，告知变化在哪几格 | `timeout_s`(1-60), `region`, `threshold`, `return_image` |
| `list_devices` | 列出所有视频设备：类型猜测（采集卡/摄像头）、有没有画面、分辨率/编码、当前用的是哪个、记住的是哪个。默认不打开摄像头 | `preview`: true 时附一张所有设备的缩略图拼图（会短暂打开摄像头） |
| `select_device` | 切换设备并记住；`"auto"` 恢复自动并清除记忆。打不开时自动退回原来的设备 | `device`（索引/名称片段/auto）, `remember`（默认 true） |
| `read_screen` | 把屏幕读成**带排版的文字**（分栏、表格、选中项、颜色），不返回图片；见下一节 | `region`, `changes_only` |

`region` 写法（都相对整屏，不随缩放变化）：`"C3"`、`"B2:D4"`、`"100,200,500,600"`（0–1000 归一化）、`""`=整屏。
以下也能被接受：`"c 3"`、`"B2-D4"`、`"B2 to D4"`、`[100,200,500,600]`、`{"x1":..}`、0–1 小数。
解析失败时错误信息里会带正确示例，方便模型自我纠正。

## 纯文字读屏 / 端侧模型（read_screen）

端侧小模型看一张截图要先过视觉编码器，一张 1280×720 的图通常要上千个 token，在低功耗 CPU 上很慢。
`read_screen` 直接把屏幕变成**带排版的文字**返回，模型不用看图：

```
SCREEN TEXT (OCR, no image) of full screen | screen 1920x1080 | captured 14:32:05.120 (age 60 ms) | OCR 0.06s (windows)
Legend: ── pane X ── = a separate column/window area; ▶ ...[selected] = selected row; [red]/[green]/[blue] = text color.
── pane A1:A2 ──
▶ 收件箱（12）[selected]
已发送
垃圾邮件
── pane B1:D2 ──
发票确认 Invoice #2041
附件是十月份的发票，金额为 12800.00 元。
付款已逾期 3 天[red]

Non-text areas (images/graphics, OCR cannot read them): F1:H2
```

**保留的版面信息**

| 屏幕上 | 文字里 |
|---|---|
| 侧边栏 / 主内容 / 多栏 | `── pane B1:D4 ──` 分栏标题（网格范围，可用 `region` 只读这一块） |
| 表格（列对齐的多行） | `── table ──`，每行一条，单元格用 ` \| ` 分隔，表头会并进表格 |
| 段落、区块间距 | 空行 |
| 同一行里隔得很开的内容（菜单、工具栏） | 三个空格 |
| 选中的列表项 | `▶ ...[selected]` |
| 按钮、标签块 | `[button]` |
| 红 / 绿 / 蓝字 | `[red]` `[green]` `[blue]` |
| 灰字（禁用、占位提示） | `[dim]` |
| 识别不确定（RapidOCR 有置信度） | `(?)` |
| 图片、图表等没有文字的区域 | 末尾 `Non-text areas: C2:D3` |

**快的原因**
- Windows 上用系统自带 OCR（原生代码、无需下载模型）；Linux / 其他系统用 RapidOCR：检测用 PP-OCRv4 mobile（自带的 v6 small 检测 1080p 要 ~8 秒，mobile 快 10 倍），在缩到 960 像素的画面上检测，识别仍用原图；关闭方向分类。
- 画面没变直接返回上次结果（毫秒级）；只变了一部分时只重新识别变化的区域。
- 服务启动时后台预热 OCR 引擎。
- `read_screen(changes_only=true)` 只返回和上次相比新增(+)/消失(-)的行，循环里反复读很省 token。

**两个引擎怎么选**（4 种合成的 1080p 中英文界面，本机 Intel Ultra 5 125H；低压 4 核大约再慢 2~3 倍）：

| 引擎 | 文字段完全正确 | 首次整屏 | 画面不变 | 适合 |
|---|---|---|---|---|
| Windows 自带 OCR（默认，Windows 上） | ~77% | ~0.07 秒 | ~7 毫秒 | 速度优先；中文为主 |
| RapidOCR（`CAPTURE_OCR_ENGINE=rapid`；Linux 默认） | ~98% | ~1.2 秒 | ~14 毫秒 | 准确优先；中英文、数字混排 |

Windows 中文识别器认英文和数字较弱（`Invoice`→`lnvoice`、`xlsx`→`又sx`），已做常见纠错（全角标点、`1乙800`→`12800`、`tO`→`to` 等）。Windows 上想要更准就装 `.[ocr-rapid]` 并设 `CAPTURE_OCR_ENGINE=rapid`。RapidOCR 首次使用会下载约 5 MB 的检测模型（安装脚本里的 `--ocr-test` 会提前下好）。

**纯文字模式**：客户端配置的 `env` 里加 `"CAPTURE_TEXT_ONLY": "1"`（或安装时用 `-TextOnly` / `--text-only`）：
不注册 `screenshot`/`zoom`，`wait_for_change` 改为"文字有变化才返回"，并直接返回变化的行；服务端说明也换成纯文字版。
不支持图片的模型应该用这个模式。

**局限**：OCR 会认错字（尤其中文识别器认英文/数字），看不到图标和图片内容；Windows 自带 OCR 需要对应语言包
（设置 → 时间和语言 → 语言 → 添加语言时勾选"光学字符识别"）。需要逐字准确的内容，多模态模型可以再用 `zoom` 核对。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `CAPTURE_DEVICE` | 空 | 强制使用某设备（索引或名称片段），优先级高于模型记住的选择。一般不用填 |
| `CAPTURE_SETTINGS_FILE` | `~/.capture-mcp/settings.json` | 记住的设备存在这里 |
| `CAPTURE_BACKEND` | auto | Windows 默认 `dshow`（MJPG 可靠、打开快），可选 `msmf` |
| `CAPTURE_WIDTH/HEIGHT/FPS` | 1920/1080/30 | 向采集卡请求的格式，实际值见 `list_devices` |
| `CAPTURE_FOURCC` | MJPG | 很多廉价卡 YUY2 下 1080p 只有 5fps 或打不开；设为空则用驱动默认 |
| `CAPTURE_MAX_SIDE` | 1280 | `detail=normal` 的长边像素；low=768，high=1920 |
| `CAPTURE_JPEG_QUALITY` | 80 | 30–95 |
| `CAPTURE_GRID` | 8x5 | 网格列x行（列用字母，最多 26） |
| `CAPTURE_TEXT_ONLY` | 0 | 1 = 纯文字模式：不提供看图工具，给不支持图片的模型用 |
| `CAPTURE_OCR_ENGINE` | auto | `auto`（Windows 用系统 OCR，否则 RapidOCR）/ `windows` / `rapid` / `none` |
| `CAPTURE_OCR_LANG` | 空 | Windows OCR 语言，如 `zh-Hans-CN`、`en-US`；空 = 已安装的中文优先 |
| `CAPTURE_OCR_SCALE` | 1.25 | Windows OCR 识别前放大倍数（实测 1.25 准确率/耗时最好） |
| `CAPTURE_OCR_THREADS` | min(4, 核数) | RapidOCR 线程数 |
| `CAPTURE_OCR_DET_SIZE` | 960 | RapidOCR 文字检测时把画面缩到的长边（0 = 原图）；实测 960 与原图准确率相同、快约一倍 |
| `CAPTURE_OCR_GRAPHICS` | 1 | 是否报告没有文字的图片/图形区域 |
| `CAPTURE_TEXT_MAX_CHARS` | 8000 | `read_screen` 输出上限，超出会截断并提示用 `region` |
| `CAPTURE_AUTOCROP` | 1 | 自动裁掉**左右/上下对称**的黑边（源分辨率比例不同时） |
| `CAPTURE_SAVE_DIR` | 空 | 把每张返回给模型的图存下来，排查"模型到底看到了什么" |
| `CAPTURE_IDLE_RELEASE_S` | 0 | 空闲 N 秒后释放设备，0=从不释放。**采集卡保持 0**：释放后另一台电脑可能认为显示器被拔掉而反复调整显示。只在用普通摄像头、或想让 OBS 等程序接手设备时才设置 |

## 给小模型的系统提示词（建议放进客户端的 system prompt）

```
你可以通过 capture-card 工具查看另一台电脑的屏幕（只能看，不能操作）。
1. 先调用 screenshot。图上有网格：列 A,B,C…，行 1,2,3…，每格左上角有标签。
2. 文字太小或不确定时，调用 zoom(region="C3") 或 zoom(region="B2:D3") 放大，不要猜。大段文字可以用 read_screen 直接读成文字（更快）。
3. 等待加载/弹窗时用 wait_for_change(timeout_s=20)，不要反复截图。
4. 工具返回 NO IMAGE / ERROR 时，如实告诉用户，不要编造屏幕内容。
5. 描述位置时用网格名（例如"右上角 G1 格的按钮"）。
6. 截图报设备错误，或画面是摄像头拍的而不是电脑屏幕时：调用 list_devices（看不出来就加 preview=true），然后 select_device 选画面是电脑屏幕的那个。
```

## 针对小模型的设计说明

| 问题 | 处理 |
|---|---|
| 1080p 被视觉编码器缩小后小字糊掉；或者动态分辨率模型 token 爆炸 | 默认长边 1280；`detail` 是枚举而非数字；`zoom` 放大局部 |
| 像素坐标理解差，缩放后对不上 | 带标签网格 + 0–1000 归一化坐标，永远相对整屏 |
| 4bit 模型 tool call 参数类型/大小写/格式出错 | schema 只用 string/number/bool；服务端宽松解析（`"10"`、`"true"`、列表、大小写都可）；错误带示例 |
| 每张图都留在上下文里，几轮就超长 | 每次只回 1 张图；`wait_for_change` 默认只回文字。若客户端支持，设置"只保留最近 N 张图" |
| 对黑屏/无信号画面编故事 | 纯色画面（含采集卡蓝屏）只返回文字 `NO IMAGE: ...`；刚打开设备的黑帧会自动重试 2.5 秒 |
| 读到旧画面 | 使用期间后台线程按帧率持续读帧（DirectShow 会扣住上次读取后的第一帧，只在请求时读就会拿到上一次截图时的旧画面）；空闲后再请求时，丢弃立即返回的缓冲帧，只接受真正等到的新帧（文字里有 age） |
| 网格标签挡住文字 | 标签半透明；`zoom` 输出不画网格 |
| 有的客户端不把工具返回的图片传给模型 | 每次都附文字元信息；用 `CAPTURE_SAVE_DIR` 排查 |

## 常见问题

- **打不开设备 / 有画面但读不到帧**：设备被 OBS、相机 App、浏览器标签页或另一个 MCP 实例占用（UVC 设备通常独占）。
- **选错设备**：让模型调用 `list_devices` / `select_device`；或者在客户端配置里写死 `CAPTURE_DEVICE`。
- **1080p 很卡**：确认 `list_devices` 里 fourcc 是 MJPG；不是的话换 `CAPTURE_BACKEND=msmf` 或降分辨率。
- **画面四周有黑边 / 坐标偏**：确认 `CAPTURE_AUTOCROP=1`；或把源电脑分辨率设为与采集分辨率同比例（16:9）。
- **首次调用慢**：首次调用才打开设备（Windows 上 1–6 秒），之后设备一直保持打开，截图很快。
- **read_screen 认错字**：OCR 的局限，中文识别器认英文/数字较弱。可以先用 `--ocr-test` 看引擎效果；Windows 上可以试 `CAPTURE_OCR_SCALE=1.5`，或装 `.[ocr-rapid]` 并设 `CAPTURE_OCR_ENGINE=rapid` 对比。
- **read_screen 说 OCR 不可用**：Windows 上装 `.[ocr]` 并确认装了 OCR 语言包；其他系统装 `.[ocr]`（RapidOCR）。
