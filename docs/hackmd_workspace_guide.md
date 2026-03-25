# Multi-Agent Workspace 使用說明

## 專案簡介

這是一個結合一般聊天模式與多代理人協作模式的 AI Chat 平台。

系統分成兩種主要模式：

- `Chat Mode`
  單一模型的一般對話模式，適合日常問答、快速測試、單輪生成。
- `Workspace Mode`
  由 `PM Agent` 主導，先釐清需求，再生成 implementation 草案與 agent team，最後進入多代理人協作執行。

---

## 系統流程

### 1. Discovery

使用者先在 `Workspace Mode` 輸入需求。

PM Agent 會：

- 用自然對話確認需求
- 補齊必要資訊
- 在資訊足夠時產出 implementation 初稿

### 2. Implementation Review

需求確認後，系統會進入 implementation 階段。

此時：

- 中央會顯示 implementation guideline 草稿
- 右側會顯示 agent roster
- 使用者可以直接修改 guideline
- 使用者可以修改每個 agent 的角色、模型、prompt
- 也可以用 `AI Generate Prompt` 補完整 prompt

### 3. Agent Configuration

按下 `Process` 後，會聚焦到右側 agent 設定區。

建議在這裡確認：

- 每個 agent 的角色是否合理
- 每個 agent 是否選到正確模型
- 雲端模型是否有填入對應 API key
- prompt 是否清楚定義職責、不可做事項、回報方式

### 4. Execution

按下 `Start Execution` 後，PM 會正式主持這一輪 execution。

執行原則：

- PM 先排好 queue
- 工程角色先做實作
- QA Engineer 會在後段逐條驗收
- 若 QA 發現問題，PM 會重新安排修正 queue
- 直到這一輪有實際成品或可交付輸出

---

## 如何使用

### 啟動前端

在 `frontend` 目錄執行：

```bash
npm install
npm run dev
```

預設網址：

```text
http://localhost:3000
```

### 啟動後端

在 `backend` 目錄執行：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

健康檢查：

```bash
curl http://127.0.0.1:8000/api/health
```

---

## 模型設定

### 本地模型

本專案支援 Ollama 本地模型，例如：

- `ollama/llama3.2`
- `ollama/qwen2.5`
- `ollama/deepseek-r1:32b`

使用前請先確認 Ollama 正在執行。

查詢模型：

```bash
curl http://127.0.0.1:11434/api/tags
```

### 雲端模型

若使用雲端模型，例如：

- `gpt-4o`
- `gemini-2.5-flash`
- `gemini-2.5-pro`

請在右上角或 agent 卡片中填入對應 API key。

注意：

- `gemini-*` 需要 Google API key
- `gpt-*` 需要 OpenAI API key

---

## Chat Mode 使用方式

`Chat Mode` 適合：

- 快速測試模型
- 一般聊天
- 驗證 local / cloud model 是否可用

使用步驟：

1. 選擇模型
2. 輸入訊息
3. 送出後等待 streaming 回覆

若模型支援 reasoning，畫面會先顯示 `Thinking...`

---

## Workspace Mode 使用方式

### Step 1. 建立新專案

切到 `Workspace Mode`，直接輸入你的需求。

例如：

```text
我要做一個賣西瓜的網站，風格要帥，其餘由你安排
```

### Step 2. 與 PM 完成需求確認

PM 會先問需求，直到資訊足夠。

若你想直接進 implementation，可以回覆：

```text
需求已確認，請生成 implementation
```

### Step 3. 修改 implementation 草案

進入 implementation review 後：

- 中央可直接修改 guideline
- 可用 AI 指令補強內容
- 確認後按 `Process`

### Step 4. 確認 agent team

在右側確認：

- role
- model
- prompt
- API key

### Step 5. 開始執行

按下 `Start Execution` 後，PM 會主持整個執行流程。

這時 agents 會：

- 按順序接手工作
- 回報實際完成內容
- 由 QA 在後段驗收
- 若有問題，PM 會安排回修

---

## 角色設計原則

### PM

- 主持 execution
- 排 queue
- 決定下一位 agent
- 遇到 QA 不通過時安排回修

### CTO / Architect

- 規劃技術拆分
- 協助 PM 排工程順序

### Frontend Developer

- 負責前端 UI、互動、狀態管理
- 應直接產出前端檔案修改

### Backend Developer

- 負責 API、資料流、串流、模型呼叫
- 應直接產出後端檔案修改

### QA Engineer

- 逐條對 guideline 驗收
- 明確指出未完成項與修正建議
- 不可用「大致可用」直接放行

---

## 實際檔案寫入規則

工程 agent 在 execution 階段會優先透過工具寫入專案檔案。

目前工具可處理：

- `write_code_file`
- `read_code_file`
- `execute_playwright_qa`

寫入位置會限制在專案 workspace 內，不會寫到專案外部。

---

## 常見問題

### 1. 為什麼 agent 只回報進度，沒有真的做事？

通常代表：

- 該角色 prompt 不夠明確
- 模型沒有正確觸發工具
- execution 沒有進入真正的工程回合

目前系統已調整為：

- 工程角色必須產出實際檔案修改
- PM 在每輪後主持下一步
- QA 最後逐條驗收

### 2. 為什麼 local model 會出現 `No generation chunks were returned`？

常見原因：

- Ollama 本身回應不穩
- 同時太多請求打到本地模型

目前已加入：

- 單線排隊
- 請求延遲
- non-stream fallback

### 3. 為什麼 implementation 沒有跳出？

通常是因為 PM 還認為需求未確認。

可直接輸入：

```text
需求已確認，請生成 implementation
```

---

## 建議使用方式

若你要做完整專案，建議流程如下：

1. 先用 `Workspace Mode`
2. 讓 PM 幫你收斂需求
3. 進 implementation review 修改草案
4. 在右側確認每個 agent 的模型與 prompt
5. 再按 `Start Execution`
6. 若中途要追加需求，直接在 execution 階段對 PM 說明

---

## 未來可持續擴充方向

- 讓 QA 直接觸發回修 loop
- 讓 agent 實作後自動產生 diff 摘要
- 讓 execution 結束時自動生成最終交付報告
- 讓前後端與 Playwright 驗證串成完整交付流程
