"use strict";
const $=id=>document.getElementById(id);
const form=$("research-form");
let current=null, busy=false, writable=false, catalog=null;
const labels={supported:"Поддерживается",unseen_molecule:"Неизвестная молекула",insufficient_target_groups:"Недостаточно групп",outside_observed_range:"Вне диапазона",sparse_process_support:"Редкая область",interval_unavailable:"Нет интервала",interval_too_wide:"Широкий интервал"};
const fmt=v=>v==null?"—":typeof v==="number"?v.toLocaleString("ru-RU",{maximumSignificantDigits:6}):String(v);
function node(tag,t,cls){const n=document.createElement(tag);if(t!=null)n.textContent=t;if(cls)n.className=cls;return n;}
function opt(value,text){const o=node("option",text);o.value=value;return o;}
function notice(text,error=false){const x=$(error?"research-error":"research-message");x.textContent=text;x.hidden=false;}
function controls(){$("refresh-research").disabled=busy;$("research-runs").disabled=busy;$("run-research").disabled=busy||!writable;$("fixture").disabled=busy||!writable;$("export-run").disabled=busy||!current;$("research-file").disabled=busy||!writable;form.querySelectorAll("input,select,textarea").forEach(e=>{e.disabled=busy||!writable||(e.name==="molecular"&&!catalog?.rdkit_available);});}
async function run(fn){if(busy)return;busy=true;$("research-error").hidden=true;$("research-message").hidden=true;controls();try{await fn();}catch(e){notice(e.message||String(e),true);}finally{busy=false;controls();}}
async function api(path,body){const r=await fetch(path,body===undefined?{cache:"no-store"}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw Error(Array.isArray(j.detail)?j.detail.map(d=>(d.loc||[]).join(".")+": "+d.msg).join("\n"):j.detail||"Ошибка API");return j;}
function table(host,heads,rows){if(!rows.length){host.replaceChildren(node("p","Нет записей.","empty-copy"));return;}const t=node("table"),thead=node("thead"),tr=node("tr");for(const h of heads){const th=node("th",h);th.scope="col";tr.append(th);}thead.append(tr);const body=node("tbody");for(const row of rows){const r=node("tr");for(const value of row)r.append(node("td",fmt(value)));body.append(r);}t.append(thead,body);host.replaceChildren(t);}
async function refresh(preferred){catalog=await api("/api/research/catalog");const auth=await api("/api/auth/me");writable=!auth.authenticated_mode||["admin","researcher"].includes(auth.user?.role);
$("research-create").hidden=!writable;
const model=$("research-model"),prev=model.value;model.replaceChildren(opt("auto","Auto — групповая CV"),...Object.entries(catalog.models).map(([k,m])=>opt(k,m.name+" · v"+m.version)));if([...model.options].some(x=>x.value===prev))model.value=prev;
$("rdkit-status").textContent=catalog.rdkit_available?"RDKit установлен. Молекулярный режим требует SMILES каждой строки.":"RDKit не установлен. Численные модели доступны; molecular extra подключается отдельно.";
const list=await api("/api/research/runs?limit=100");
const select=$("research-runs");select.replaceChildren(opt("","—"),...list.items.map(x=>opt(x.id,x.name+" · "+x.model+(x.dataset_kind==="synthetic_fixture"?" · ФИКСТУРА":""))));
$("registry-caption").textContent=list.total+" расчётов / "+catalog.max_runs+". Общий реестр организации.";
const id=list.items.some(x=>x.id===preferred)?preferred:list.items[0]?.id;
if(id){select.value=id;await load(id);}else{current=null;$("research-result").hidden=true;}
}
async function load(id){current=await api("/api/research/runs/"+encodeURIComponent(id));render();}
const svg=(tag,attributes,text)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const [k,v] of Object.entries(attributes||{}))e.setAttribute(k,String(v));if(text!=null)e.textContent=text;return e;};
function scatter(){if(!current)return;const target=$("scatter-target").value,d=current.result.validation[target];const host=$("research-scatter");if(!d?.diagnostic_scatter?.length){host.replaceChildren(node("p","Нет данных проверки.","empty-copy"));return;}
const rows=d.diagnostic_scatter,values=rows.flatMap(p=>[p.observed,p.predicted]);let low=Math.min(...values),high=Math.max(...values);const pad=(high-low||Math.max(Math.abs(low),1))*.08;low-=pad;high+=pad;
const W=640,H=290,L=88,R=18,T=16,B=45;const x=v=>L+(v-low)/(high-low)*(W-L-R),y=v=>H-B-(v-low)/(high-low)*(H-T-B);
const s=svg("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Факт и диагностический прогноз "+target});s.append(svg("title",{},target));
for(let j=0;j<5;j++){const v=low+(high-low)*j/4;s.append(svg("line",{x1:L,x2:W-R,y1:y(v),y2:y(v),class:"grid"}),svg("text",{x:L-7,y:y(v)+4,"text-anchor":"end"},fmt(v)),svg("text",{x:x(v),y:H-B+17,"text-anchor":"middle"},fmt(v)));}
s.append(svg("line",{x1:x(low),y1:y(low),x2:x(high),y2:y(high),stroke:"#83988b","stroke-dasharray":"5 4"}));for(const p of rows){const q=svg("circle",{cx:x(p.observed),cy:y(p.predicted),r:4,fill:p.supported?"#23694e":"#a18a69",opacity:.85});q.append(svg("title",{},"Факт "+fmt(p.observed)+"; прогноз "+fmt(p.predicted)+(p.supported?" · поддерживается":" · отказ")));s.append(q);}
s.append(svg("text",{x:W-R,y:H-7,"text-anchor":"end"},"Факт"),svg("text",{x:L,y:10},"Прогноз"));host.replaceChildren(s);}
function render(){const d=current.result,c=current.input;$("research-result").hidden=false;$("research-kind").textContent=c.dataset_kind==="synthetic_fixture"?"ПРОГРАММНАЯ ФИКСТУРА — НЕ ФИЗИЧЕСКИЕ ДАННЫЕ":"РЕАЛЬНЫЕ ДАННЫЕ (ЗАЯВЛЕНО ОПЕРАТОРОМ)";$("research-title").textContent=c.name;$("research-version").textContent="Модель "+d.selected_model+" · "+current.model_version.slice(0,16)+" · "+current.created_at;
$("research-stats").replaceChildren(...Object.entries(d.counts).map(([k,v])=>{const h=node("div",null,"stat");h.append(node("span",({train:"Обучение",calibration:"Калибровка",validation:"Проверка"})[k]),node("strong",fmt(v.groups)),node("small",v.rows+" наблюдений"));return h;}));
table($("selection-table"),["Модель","Ошибка","Выбор"],d.selection.map(m=>[m.model,fmt(m.normalized_group_mae),m.model===d.selected_model?"Выбрана":"—"]));
table($("calibration-table"),["Цель","Группы","Мин. групп","Квантиль"],d.calibration.map(v=>[v.target,v.groups,v.minimum_finite_groups,v.quantile]));
table($("validation-table"),["Цель","Группы","Macro MAE","Покрытие групп","Принято / наблюдений"],Object.entries(d.validation).map(([k,v])=>[k,v.groups,v.group_macro_mae,v.whole_group_coverage==null?"—":fmt(v.whole_group_coverage*100)+"%",v.accepted_rows+" / "+v.observed_rows]));
const select=$("scatter-target"),old=select.value;select.replaceChildren(...Object.keys(d.validation).map(x=>opt(x,x)));if(Object.hasOwn(d.validation,old))select.value=old;scatter();
table($("candidate-table"),["Кандидат","Свойство","Прогноз","Интервал","Статус"],d.candidate_predictions.flatMap(q=>Object.entries(q.properties).map(([k,v])=>[q.id,k,v.prediction,v.interval?v.interval.map(fmt).join(" … "):"—",labels[v.reason]||v.reason])));
$("collection-summary").textContent="Затраты "+fmt(d.collection_plan.spent_minor/100)+" "+c.currency+" · остаток "+fmt(d.collection_plan.remaining_minor/100)+" "+c.currency;
table($("collection-table"),["Кандидат","Стоимость","Метод","Статус"],d.collection_plan.suggestions.map(v=>[v.id,fmt(v.cost_minor/100)+" "+c.currency,v.method,"Нужно согласование"]));
$("research-manifest").textContent=JSON.stringify({input_sha256:current.input_sha256,model_version:current.model_version,record_sha256:current.record_sha256,runtime:current.runtime,provenance:c.provenance},null,2);controls();}
function example(){const e=form.elements;e.name.value="Тестовый пример — не химические измерения";e.dataset_kind.value="synthetic_fixture";e.provenance.value="Искусственная программная фикстура; каждая уникальная x — условная независимая группа. Не физические данные.";e.config.value=JSON.stringify({factors:[{name:"x",lower:0,upper:1,unit:"a.u."}],targets:[{name:"property",lower:.6,scale:1,unit:"a.u."}],mode:"real"},null,2);e.widths.value=JSON.stringify({property:.2},null,2);e.model.value="auto";e.coverage.value=".9";e.molecular.checked=e.coupled_mass.checked=false;
const groups={train:[.05,.15,.25,.35,.65,.75,.85,.95],calibration:[.1,.2,.3,.4,.5,.6,.7,.8,.9],validation:[.12,.42,.82]};
e.csv.value="group_id,role,x,property\n"+Object.entries(groups).flatMap(([r,vs])=>vs.map((x,i)=>r+"-"+i+","+r+","+x+","+(.2+.6*x).toFixed(5))).join("\n")+"\n";
e.candidates.value=JSON.stringify([{id:"fixture-A",parameters:{x:.45},cost_minor:500000},{id:"fixture-B",parameters:{x:.99},cost_minor:500000},{id:"fixture-C",parameters:{x:.55},cost_minor:700000}],null,2);e.reviewed.checked=true;e.currency.value="RUB";e.budget.value="1000000";e.limit.value="3";}
$("fixture").addEventListener("click",()=>{if(!busy&&writable)example();});
$("refresh-research").addEventListener("click",()=>run(()=>refresh(current?.id)));
$("research-runs").addEventListener("change",()=>run(async()=>{const id=$("research-runs").value;if(id)await load(id);else{current=null;$("research-result").hidden=true;}}));
$("scatter-target").addEventListener("change",scatter);
$("research-file").addEventListener("change",()=>run(async()=>{const f=$("research-file").files[0];if(!f)return;if(f.size>200000)throw Error("Файл превышает 200 КБ");form.elements.csv.value=await f.text();notice("CSV загружен. Проверка схемы выполняется сервером.");}));
form.addEventListener("submit",e=>{e.preventDefault();if(!writable)return;run(async()=>{const v=form.elements;const payload={name:v.name.value,dataset_kind:v.dataset_kind.value,provenance:v.provenance.value,config:JSON.parse(v.config.value),csv:v.csv.value,model:v.model.value,molecular:v.molecular.checked,coupled_mass:v.coupled_mass.checked,coverage:Number(v.coverage.value),max_interval_width:JSON.parse(v.widths.value),currency:v.currency.value,budget_minor:Number(v.budget.value),max_suggestions:Number(v.limit.value),candidate_pool_reviewed:v.reviewed.checked,candidates:JSON.parse(v.candidates.value)};const result=await api("/api/research/runs",payload);await refresh(result.id);$("research-create").open=false;notice("Исследовательский расчёт сохранён.");});});
$("export-run").addEventListener("click",()=>{if(!current||busy)return;const url=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:"application/json"}));const a=node("a");a.href=url;a.download="polysearch-research-"+current.id+".json";document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
run(()=>refresh());
