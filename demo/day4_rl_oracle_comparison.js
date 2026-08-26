let data, step = 0, timer = null;
const intervalMs = 180;
const definitions = [
  ['0', 'UNTRAINED', 'Prior RL training: 0 placements'],
  ['750', 'PARTIALLY TRAINED', 'Prior RL training: 750 placements'],
  ['3000', 'MORE TRAINED', 'Prior RL training: 3,000 placements'],
  ['oracle', 'ORACLE', 'Fixed Day 2 reference policy'],
];
const panels = document.querySelector('#panels');

function cells(frame) {
  return frame.board.join('').split('').map(value =>
    `<i class="cell ${value === '1' ? 'on' : ''}"></i>`).join('');
}
function render() {
  const last = Math.max(...definitions.map(([key]) => data.policies[key].frames.length - 1));
  step = Math.min(step, last);
  panels.innerHTML = definitions.map(([key, title, descriptor]) => {
    const policy = data.policies[key];
    const index = Math.min(step, policy.frames.length - 1);
    const frame = policy.frames[index];
    const done = index === policy.frames.length - 1;
    const status = done ? (policy.status === 'cap'
      ? 'STILL ALIVE — TEST STOPPED AT 300' : 'GAME OVER') : '';
    return `<section class="panel ${done ? 'done' : ''}"><h2>${title}</h2><div class="descriptor">${descriptor}</div><div class="board ${frame.cleared_now ? 'clear' : ''}">${cells(frame)}</div><div class="stats"><span>Test pieces placed: <b>${frame.placements}</b></span><span>Lines cleared: <b>${frame.lines}</b></span></div><div class="status">${status}</div></section>`;
  }).join('');
  document.querySelector('#step').textContent = `Shared test piece ${step} / ${last}`;
}
function pause() { if (timer) clearInterval(timer); timer = null; }
function start() { pause(); timer = setInterval(() => { const last = Math.max(...definitions.map(([key]) => data.policies[key].frames.length - 1)); if (step >= last) return pause(); step += 1; render(); }, intervalMs); }
document.querySelector('#start').onclick = start;
document.querySelector('#pause').onclick = pause;
document.querySelector('#replay').onclick = () => { pause(); step = 0; render(); };
fetch('day4_rl_oracle_comparison.json').then(response => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then(value => {
  data = value;
  document.querySelector('#provenance').textContent = `Held-out seed ${data.seed} · 300-piece test cap · frozen Day 4 learner snapshots · ${data.policies.oracle.expert_id}, beam width ${data.policies.oracle.beam_width}`;
  render();
}).catch(error => { panels.innerHTML = `<p>Could not load replay data: ${error}. Serve the repository through a local HTTP server.</p>`; });
