# Fast Setup — Ollama on Remote + OpenCode

Get Ollama running on a remote server and connect your local OpenCode to it in ~5 minutes.

---

## Prerequisites

- SSH access to a remote machine with a GPU
- Local machine has `ssh`, `ollama`, and [opencode.ai](https://opencode.ai) installed

---

## On the remote server

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull a model

```bash
ollama pull qwen3.6:35b-a3b
```

Ollama listens on `localhost:11434` by default.

---

## On your local machine

### 1. Clone this repo and set up `.env`

```bash
git clone <your-repo-url>
cd llm-playground
cp .env.example .env
```

Edit `.env`:

```env
REMOTE_HOST=your-remote-server.edu
KERB_PRINCIPAL=youruser@YOUR.INSTITUTION.EDU
```

- `REMOTE_HOST` — the SSH hostname of your remote GPU server
- `KERB_PRINCIPAL` — your Kerberos principal (e.g. `jdoe@MIT.EDU`)

### 2. Add SSH config for Ollama

Add this to `~/.ssh/config`:

```ssh-config
Host <REMOTE_HOST>-ollama
    HostName <your-remote-server-ip-or-hostname>
    User <your-username>
    LocalForward 11435 localhost:11434
```

Replace `<REMOTE_HOST>` and the host details with your actual values. The `LocalForward` line tunnels `localhost:11435` → `<remote>:11434` so Ollama is reachable locally through SSH.

> If your SSH setup uses a jump host, add `ProxyJump <jump-host>` between `Host` and `HostName`.

### 3. Open the tunnel

```bash
./connect-remote.sh
```

This starts the SSH port-forward (keeps running in the foreground).

### 4. Configure OpenCode

Create `opencode.json` in your project directory (or `~/.config/opencode/opencode.json` for a global config):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "name": "Ollama on remote",
      "options": {
        "baseURL": "http://127.0.0.1:11435/v1"
      },
      "models": {
        "qwen3.6:35b-a3b": {
          "name": "Qwen3.6 35B",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  }
}
```

### 5. Run OpenCode

```bash
opencode
```

Select **Ollama on remote › Qwen3.6 35B** from the model picker.

---

## Quick reference

| Command | Purpose |
|---|---|
| `./connect-remote.sh` | Open SSH tunnel to remote Ollama |
| `./connect-remote-llm.sh` | Open SSH tunnel for llama.cpp server |
| `ollama-remote list` | List models on remote Ollama |
| `ollama-remote pull qwen3.6:35b-a3b` | Pull a model to remote Ollama |
| `ollama-remote-stop` | Close the SSH tunnel |

> **Note:** Ollama on the remote server is convenient but slower than llama.cpp. Use it for quick iteration; switch to llama.cpp for performance-critical work.
