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
- 可选 OCR：`pip install -e .[ocr]`（rapidocr + onnxruntime；旧版 `rapidocr-onnxruntime` 也支持），装了才会出现 `read_text` 工具。
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
| `read_text` | OCR，返回 `[格子] 文字 (坐标)`（需安装 rapidocr） | `region` |

`region` 写法（都相对整屏，不随缩放变化）：`"C3"`、`"B2:D4"`、`"100,200,500,600"`（0–1000 归一化）、`""`=整屏。
以下也能被接受：`"c 3"`、`"B2-D4"`、`"B2 to D4"`、`[100,200,500,600]`、`{"x1":..}`、0–1 小数。
解析失败时错误信息里会带正确示例，方便模型自我纠正。

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
| `CAPTURE_AUTOCROP` | 1 | 自动裁掉**左右/上下对称**的黑边（源分辨率比例不同时） |
| `CAPTURE_SAVE_DIR` | 空 | 把每张返回给模型的图存下来，排查"模型到底看到了什么" |
| `CAPTURE_IDLE_RELEASE_S` | 0 | 空闲 N 秒后释放设备（让 OBS 等程序能用），0=不释放 |

## 给小模型的系统提示词（建议放进客户端的 system prompt）

```
你可以通过 capture-card 工具查看另一台电脑的屏幕（只能看，不能操作）。
1. 先调用 screenshot。图上有网格：列 A,B,C…，行 1,2,3…，每格左上角有标签。
2. 文字太小或不确定时，调用 zoom(region="C3") 或 zoom(region="B2:D3") 放大，不要猜。需要逐字准确时用 read_text。
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
| 读到旧画面 | 后台线程持续 grab 清空驱动缓冲，只在请求时解码；保证返回的帧是请求之后抓的（文字里有 age） |
| 网格标签挡住文字 | 标签半透明；`zoom` 输出不画网格 |
| 有的客户端不把工具返回的图片传给模型 | 每次都附文字元信息；用 `CAPTURE_SAVE_DIR` 排查 |

## 常见问题

- **打不开设备 / 有画面但读不到帧**：设备被 OBS、相机 App、浏览器标签页或另一个 MCP 实例占用（UVC 设备通常独占）。
- **选错设备**：让模型调用 `list_devices` / `select_device`；或者在客户端配置里写死 `CAPTURE_DEVICE`。
- **1080p 很卡**：确认 `list_devices` 里 fourcc 是 MJPG；不是的话换 `CAPTURE_BACKEND=msmf` 或降分辨率。
- **画面四周有黑边 / 坐标偏**：确认 `CAPTURE_AUTOCROP=1`；或把源电脑分辨率设为与采集分辨率同比例（16:9）。
- **首次调用慢**：首次调用才打开设备（Windows 上 2–6 秒），之后很快。
- **read_text 慢**：rapidocr 在 CPU 上每次约 5–10 秒（实测与图片大小关系不大），第一次还要加载模型。它是可选的，平时优先用 zoom；只在需要逐字准确（数字、代码）时用。
