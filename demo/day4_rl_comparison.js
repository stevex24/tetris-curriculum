let data, step = 0, timer = null, clearTimer = null;
const exposures = [0, 750, 3000];
const titles = {0: 'Untrained', 750: 'RL trained', 3000: 'RL trained'};
const panels = document.querySelector('#panels');

function board(frame) {
  return frame.board.join('').split('').map(value =>
    `<i class="cell ${value === '1' ? 'on' : ''}"></i>`).join('');
}

function render() {
  const maximum = Math.max(...exposures.map(x => data.policies[x].frames.length - 1));
  step = Math.min(step, maximum);
  panels.innerHTML = exposures.map(exposure => {
    const policy = data.policies[exposure];
    const index = Math.min(step, policy.frames.length - 1);
    const frame = policy.frames[index];
    const done = index === policy.frames.length - 1;
    const label = done
      ? `${policy.status === 'game_over' ? 'GAME OVER' : 'DEMO CAP'} — ${policy.placements} placements / ${policy.lines} lines`
      : `Placement ${frame.placements}`;
    return `<section class="panel ${done ? 'done' : ''}"><h2>${titles[exposure]} — exposure ${exposure}</h2><div class="exposure">Exposure: ${exposure}</div><div id="board-${exposure}" class="board ${frame.cleared_now ? 'clear' : ''}">${board(frame)}</div><div class="stats"><span>Pieces: <b>${frame.placements}</b></span><span>Lines: <b>${frame.lines}</b></span></div><div class="piece">${frame.piece ? `Piece ${frame.piece} · rotation ${frame.action[0]} · x ${frame.action[1]}` : 'Ready'}</div><div class="status">${label}</div></section>`;
  }).join('');
  document.querySelector('#step').textContent = `Shared step ${step} / ${maximum}`;
}

function pause() { if (timer) clearInterval(timer); timer = null; }
function start() { pause(); timer = setInterval(() => { const last = Math.max(...exposures.map(x => data.policies[x].frames.length - 1)); if (step >= last) return pause(); step += 1; render(); }, 180); }
document.querySelector('#start').onclick = start;
document.querySelector('#pause').onclick = pause;
document.querySelector('#replay').onclick = () => { pause(); step = 0; render(); };
document.querySelector('#next').onclick = () => { pause(); step += 1; render(); };

fetch('day4_rl_comparison.json').then(response => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then(value => {
  data = value;
  document.querySelector('#provenance').textContent = `Held-out seed ${data.seed} · identical ${data.demo_cap}-piece stream · frozen checkpoints from the committed Day 4 result · no expert in replay`;
  render();
}).catch(error => { panels.innerHTML = `<p>Could not load replay data: ${error}. Serve the repository through a local HTTP server.</p>`; });
