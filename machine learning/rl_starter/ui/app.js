const taskEl = document.getElementById("task");
const gameEl = document.getElementById("game");
const trainFields = document.getElementById("train-fields");
const evalFields = document.getElementById("eval-fields");
const logsEl = document.getElementById("logs");
const stateEl = document.getElementById("state");
const cmdEl = document.getElementById("cmd");
const modelSelect = document.getElementById("model_path");
const runNameEl = document.getElementById("run_name");

const runBtn = document.getElementById("run");
const stopBtn = document.getElementById("stop");

function toggleFields() {
  const isTrain = taskEl.value === "train";
  trainFields.classList.toggle("hidden", !isTrain);
  evalFields.classList.toggle("hidden", isTrain);
}

taskEl.addEventListener("change", toggleFields);
gameEl.addEventListener("change", () => {
  const defaultName = gameEl.value === "pong" ? "pong_dqn" : "lunarlander_ppo";
  if (!runNameEl.value || runNameEl.value === "pong_dqn" || runNameEl.value === "lunarlander_ppo") {
    runNameEl.value = defaultName;
  }
  refreshModels();
});

toggleFields();

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

async function refreshStatus() {
  const res = await fetch("/status");
  const data = await res.json();
  stateEl.textContent = data.state;
  cmdEl.textContent = data.command || "-";
}

async function refreshLogs() {
  const res = await fetch("/logs");
  const data = await res.json();
  logsEl.textContent = data.lines.join("\n");
  logsEl.scrollTop = logsEl.scrollHeight;
}

async function refreshModels() {
  const game = gameEl.value;
  const res = await fetch(`/models?game=${encodeURIComponent(game)}`);
  const data = await res.json();
  const current = modelSelect.value;
  modelSelect.innerHTML = "";
  data.files.forEach((file) => {
    const opt = document.createElement("option");
    opt.value = file;
    opt.textContent = file;
    modelSelect.appendChild(opt);
  });
  if (current) {
    modelSelect.value = current;
  }
}

runBtn.addEventListener("click", async () => {
  const isTrain = taskEl.value === "train";
  const payload = {
    task: taskEl.value,
    game: gameEl.value,
    run_name: runNameEl.value.trim() || `${gameEl.value}_run`,
    timesteps: Number(document.getElementById("timesteps").value || 0),
    n_envs: Number(document.getElementById("n_envs").value || 1),
    device: document.getElementById("device").value,
    vec_env: document.getElementById("vec_env").value,
    vec_normalize: document.getElementById("vec_normalize").checked,
    episodes: Number(document.getElementById("episodes").value || 1),
    eval_device: document.getElementById("eval_device").value,
    model_path: modelSelect.value ? `models/${modelSelect.value}` : undefined,
  };
  if (!isTrain) {
    payload.device = payload.eval_device;
  }
  await postJson("/run", payload);
  await refreshStatus();
});

stopBtn.addEventListener("click", async () => {
  await postJson("/stop", {});
  await refreshStatus();
});

setInterval(refreshStatus, 1000);
setInterval(refreshLogs, 1000);
setInterval(refreshModels, 5000);

refreshStatus();
refreshLogs();
refreshModels();
