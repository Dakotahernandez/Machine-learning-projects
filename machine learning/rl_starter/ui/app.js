const taskEl = document.getElementById("task");
const gameEl = document.getElementById("game");
const trainFields = document.getElementById("train-fields");
const evalFields = document.getElementById("eval-fields");
const logsEl = document.getElementById("logs");
const stateEl = document.getElementById("state");
const cmdEl = document.getElementById("cmd");

const runBtn = document.getElementById("run");
const stopBtn = document.getElementById("stop");

function toggleFields() {
  const isTrain = taskEl.value === "train";
  trainFields.classList.toggle("hidden", !isTrain);
  evalFields.classList.toggle("hidden", isTrain);
}

taskEl.addEventListener("change", toggleFields);

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

runBtn.addEventListener("click", async () => {
  const isTrain = taskEl.value === "train";
  const payload = {
    task: taskEl.value,
    game: gameEl.value,
    timesteps: Number(document.getElementById("timesteps").value || 0),
    n_envs: Number(document.getElementById("n_envs").value || 1),
    device: document.getElementById("device").value,
    vec_env: document.getElementById("vec_env").value,
    vec_normalize: document.getElementById("vec_normalize").checked,
    episodes: Number(document.getElementById("episodes").value || 1),
    eval_device: document.getElementById("eval_device").value,
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

refreshStatus();
refreshLogs();
