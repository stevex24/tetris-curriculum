let data=null, step=0, timer=null;
const panels=document.querySelector('#panels');
const stop=()=>{if(timer){clearInterval(timer);timer=null;}};
function cells(frame){return frame.board.join('').split('').map(v=>`<i class="cell ${v==='1'?'on':''}"></i>`).join('');}
function render(){
  if(!data)return;
  const max=Math.max(...data.policy_order.map(k=>data.policies[k].frames.length-1));
  step=Math.min(step,max);
  panels.innerHTML=data.policy_order.map(name=>{
    const p=data.policies[name], i=Math.min(step,p.frames.length-1), f=p.frames[i], done=i===p.frames.length-1;
    const status=done?(p.status==='game_over'?'GAME OVER':'DEMO CAP'):`Step ${step}`;
    return `<section class="panel ${done?'done':''}"><h2>${p.label}</h2><div class="descriptor">Frozen Phase 2 policy</div><div class="rating">Simulator rating ${p.rating.toFixed(1)} · gain ${p.rating_gain>=0?'+':''}${p.rating_gain.toFixed(1)}</div><div class="board ${f.cleared_now?'clear':''}">${cells(f)}</div><div class="stats"><span>Placements: <b>${f.placements}</b></span><span>Lines: <b>${f.lines}</b></span></div><div class="status">${status} · cap ${data.cap}</div></section>`;
  }).join('');
  document.querySelector('#position').textContent=`Shared step ${step} / ${max}`;
}
function play(){stop();timer=setInterval(()=>{const max=Math.max(...data.policy_order.map(k=>data.policies[k].frames.length-1));if(step>=max){stop();return;}step+=1;render();},180);}
document.querySelector('#play').onclick=play; document.querySelector('#pause').onclick=stop;
document.querySelector('#restart').onclick=()=>{stop();step=0;render();};
document.querySelector('#step').onclick=()=>{stop();step+=1;render();};
fetch('phase2_comparison.json').then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();}).then(v=>{data=v;document.querySelector('#provenance').textContent=`Frozen source: ${data.integrity.source} · replicate ${data.replicate} · first configured evaluation seed ${data.seed} · stream SHA-256 ${data.piece_stream_sha256}`;render();}).catch(e=>{panels.innerHTML=`<p>Could not load replay data: ${e}. Serve the repository through a local HTTP server.</p>`;});
