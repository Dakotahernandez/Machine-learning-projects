const taskEl = document.getElementById("task");
const gameEl = document.getElementById("game");
const trainFields = document.getElementById("train-fields");
const evalFields = document.getElementById("eval-fields");
const logsEl = document.getElementById("logs");
const stateEl = document.getElementById("state");
const cmdEl = document.getElementById("cmd");
const modelSelect = document.getElementById("model_path");
const runNameEl = document.getElementById("run_name");
const nEnvsEl = document.getElementById("n_envs");
const doneBanner = document.getElementById("done-banner");
const vecEnvRow = document.getElementById("row-vec-env");
const vecNormRow = document.getElementById("row-vec-norm");

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
  if (gameEl.value === "pong") {
    nEnvsEl.value = "1";
  }
  const isLunar = gameEl.value === "lunarlander";
  vecEnvRow.classList.toggle("hidden", !isLunar);
  vecNormRow.classList.toggle("hidden", !isLunar);
  refreshModels();
});

toggleFields();
const initialIsLunar = gameEl.value === "lunarlander";
vecEnvRow.classList.toggle("hidden", !initialIsLunar);
vecNormRow.classList.toggle("hidden", !initialIsLunar);

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

  stateEl.className = "";
  doneBanner.classList.add("hidden");
  doneBanner.classList.remove("ok", "err");

  if (data.state === "running") {
    stateEl.classList.add("running");
  } else if (data.state === "idle") {
    stateEl.classList.add("idle");
  } else if (data.state.startsWith("exit(")) {
    const code = data.state.slice(5, -1);
    if (code === "0") {
      stateEl.classList.add("exit-ok");
      doneBanner.textContent = "Done: run completed successfully.";
      doneBanner.classList.add("ok");
      doneBanner.classList.remove("hidden");
    } else {
      stateEl.classList.add("exit-error");
      doneBanner.textContent = `Done with errors (exit ${code}). See logs for details.`;
      doneBanner.classList.add("err");
      doneBanner.classList.remove("hidden");
    }
  }
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
    verbose: Number(document.getElementById("verbose").value || 0),
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
